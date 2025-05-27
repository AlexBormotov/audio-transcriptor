#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Программа для онлайн-распознавания речи с микрофона с использованием Whisper.
Распознанный текст выводится в консоль в режиме реального времени и сохраняется в файл.
"""

import os
import time
import wave
import sys
import pyaudio
import numpy as np
import torch
from datetime import datetime
from faster_whisper import WhisperModel
import threading
import re

# Константы для записи аудио
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Частота дискретизации для Whisper
WINDOW_SIZE_SECONDS = 5  # Размер скользящего окна в секундах

# Создание папки для сохранения транскрипций
OUTPUT_DIR = "transcriptions"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def record_chunk(p, stream, file_path, chunk_length=1):
    """
    Записывает аудио чанк из потока и сохраняет его в WAV файл.
    
    Args:
        p: Экземпляр PyAudio
        stream: Аудио поток
        file_path: Путь для сохранения WAV файла
        chunk_length: Длина чанка в секундах
    
    Returns:
        Список записанных фреймов
    """
    frames = []
    try:
        for _ in range(0, int(16000 / 1024 * chunk_length)):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
        
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
        wf.close()
    except Exception as e:
        print(f"\nОшибка при записи аудио: {e}")
    
    return frames

def transcribe_chunk(model, chunk_file):
    """
    Распознает речь в аудио файле.
    
    Args:
        model: Модель Whisper
        chunk_file: Путь к аудио файлу
    
    Returns:
        Распознанный текст
    """
    segments, _ = model.transcribe(chunk_file, beam_size=5)
    text = " ".join([segment.text for segment in segments])
    return text

def get_default_input_device():
    """Возвращает индекс устройства ввода по умолчанию"""
    p = pyaudio.PyAudio()
    try:
        default_device = p.get_default_input_device_info()
        p.terminate()
        return default_device['index']
    except Exception as e:
        print(f"Ошибка при получении устройства ввода: {e}")
        p.terminate()
        # Выводим список доступных устройств ввода
        print("Доступные устройства ввода:")
        info = p.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        for i in range(0, num_devices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print(f"ID: {i}, Имя: {p.get_device_info_by_host_api_device_index(0, i).get('name')}")
        p.terminate()
        sys.exit(1)

def save_full_audio(frames_buffer, wav_path, p):
    """Сохраняет весь аудиобуфер в WAV-файл."""
    wf = wave.open(wav_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(RATE)
    for frames in frames_buffer:
        wf.writeframes(b''.join(frames))
    wf.close()

def format_text_with_newlines(text):
    """Разбивает текст на предложения и делает новую строку после точки, !, ?"""
    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text.strip())
    # Разбиваем по знакам препинания
    sentences = re.split(r'([.!?])', text)
    result = ''
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i].strip()
        mark = sentences[i+1]
        if sentence:
            result += sentence + mark + '\n'
    # Добавляем остаток, если есть
    if len(sentences) % 2 != 0 and sentences[-1].strip():
        result += sentences[-1].strip()
    return result

def recognize_and_save(model, wav_path, output_file, stop_state):
    """Распознаёт весь WAV-файл и перезаписывает текстовый файл. Проверяет слово 'стоп'."""
    try:
        segments, _ = model.transcribe(wav_path, beam_size=5)
        text = " ".join([segment.text for segment in segments]).strip()
        formatted = format_text_with_newlines(text)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=== Начало транскрибации ===\n\n")
            f.write(formatted + "\n")
        # Проверяем наличие слова "стоп" или "stop" среди последних 10 слов
        words = re.findall(r'\w+', text.lower())
        last_words = words[-10:] if len(words) >= 10 else words
        found_stop = any(w in ["стоп", "stop"] for w in last_words)
        # Если найдено слово стоп/stop и текст не меняется — запускаем таймер
        if found_stop:
            if stop_state.get('last_text') == text:
                # Если текст не меняется, увеличиваем время ожидания
                if stop_state.get('stop_time') is None:
                    stop_state['stop_time'] = time.time()
            else:
                # Если текст изменился, сбрасываем таймер
                stop_state['stop_time'] = None
            stop_state['last_text'] = text
        else:
            stop_state['stop_time'] = None
            stop_state['last_text'] = text
    except Exception as e:
        print(f"\n[ОШИБКА] Не удалось распознать полный WAV: {e}")

def periodic_recognition(frames_buffer, wav_path, output_file, model, p, stop_event, stop_state):
    """Поток для периодического распознавания всей записи и обновления файла. Останавливает по слову 'стоп'."""
    while not stop_event.is_set():
        save_full_audio(frames_buffer, wav_path, p)
        recognize_and_save(model, wav_path, output_file, stop_state)
        # Если был сказан "стоп" и прошло 2 секунды молчания — останавливаем
        if stop_state.get('stop_time') and (time.time() - stop_state['stop_time'] > 2):
            print("\n[INFO] Обнаружено слово 'стоп' и пауза. Останавливаю запись.")
            stop_event.set()
            break
        stop_event.wait(1)

def main():
    # Выбор модели Whisper
    model_size = "base"  # Мультиязычная модель. Можно изменить на "tiny", "base", "small", "medium", "large"

    # Принудительно использовать CUDA
    if not torch.cuda.is_available():
        print("[ОШИБКА] CUDA не обнаружена! Проверьте, что драйверы Nvidia и CUDA Toolkit установлены, и видеокарта поддерживается.")
        print("Скрипт завершён. Для работы на GPU необходима поддержка CUDA.")
        sys.exit(1)
    device = "cuda"
    compute_type = "float16"  # Оптимально для RTX 4070
    
    # Инициализация модели Whisper
    print(f"Загрузка модели Whisper {model_size}...")
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(f"Модель загружена на устройстве {device}!")
    except Exception as e:
        print(f"Ошибка при загрузке модели Whisper: {e}")
        print("Попробуйте изменить модель на 'tiny.en' вместо 'base.en'")
        sys.exit(1)
    
    # Инициализация PyAudio и открытие потока с микрофона по умолчанию
    p = pyaudio.PyAudio()
    try:
        default_device_index = get_default_input_device()
        stream = p.open(format=FORMAT, 
                        channels=CHANNELS, 
                        rate=RATE, 
                        input=True, 
                        frames_per_buffer=CHUNK,
                        input_device_index=default_device_index)
    except Exception as e:
        print(f"Ошибка при открытии потока с микрофона: {e}")
        p.terminate()
        sys.exit(1)
    
    # Создаем файл для записи транскрибации до начала записи
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(OUTPUT_DIR, f"transcription_{timestamp}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== Начало транскрибации ===\n\n")
    print(f"Файл для транскрибации создан: {output_file}")
    print("Начинаю запись с микрофона. Нажмите Ctrl+C для остановки...")

    # Буфер для хранения всех аудиофреймов
    frames_buffer = []
    # Для каждого чанка будем хранить список фреймов
    # Например: frames_buffer = [[chunk1_frames], [chunk2_frames], ...]

    # Путь к полному WAV-файлу
    full_wav_path = os.path.join(OUTPUT_DIR, f"full_record_{timestamp}.wav")

    # Событие для остановки фонового потока
    stop_event = threading.Event()
    # Состояние для отслеживания "стопа"
    stop_state = {'stop_time': None}
    # Запускаем фоновый поток для периодического распознавания
    recognition_thread = threading.Thread(
        target=periodic_recognition,
        args=(frames_buffer, full_wav_path, output_file, model, p, stop_event, stop_state),
        daemon=True
    )
    recognition_thread.start()

    try:
        chunk_counter = 0
        while not stop_event.is_set():
            chunk_counter += 1
            temp_file = "temp_chunk.wav"
            chunk_frames = record_chunk(p, stream, temp_file)
            frames_buffer.append(chunk_frames)
            # Удаляем временный файл
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"\nОшибка при удалении временного файла: {e}")
    except KeyboardInterrupt:
        print("\nОстановка записи...")
    finally:
        stop_event.set()
        recognition_thread.join()
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        p.terminate()
        # Финальное обновление файла
        save_full_audio(frames_buffer, full_wav_path, p)
        recognize_and_save(model, full_wav_path, output_file, stop_state)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write("\n=== Конец транскрибации ===\n")
        print(f"\nПолная транскрипция сохранена в {output_file}")

if __name__ == "__main__":
    main() 