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
    
    # Буфер для хранения аудио данных скользящего окна
    window_buffer = []
    # Максимальный размер буфера в чанках
    max_buffer_chunks = int(WINDOW_SIZE_SECONDS / 1)  # Размер окна / размер чанка
    
    # Переменная для накопления всего текста
    accumulated_transcription = ""
    
    # Создаем файл для записи транскрибации до начала записи
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(OUTPUT_DIR, f"transcription_{timestamp}.txt")
    
    # Создаем пустой файл
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== Начало транскрибации ===\n\n")
    
    print(f"Файл для транскрибации создан: {output_file}")
    print("Начинаю запись с микрофона. Нажмите Ctrl+C для остановки...")
    
    # Переменная для отслеживания последней записанной транскрибации
    last_saved_transcription = ""
    
    try:
        chunk_counter = 0
        while True:
            chunk_counter += 1
            
            # Имя временного файла
            temp_file = "temp_chunk.wav"
            
            # Запись чанка
            chunk_frames = record_chunk(p, stream, temp_file)
            
            # Добавление нового чанка в буфер скользящего окна
            window_buffer.append(chunk_frames)
            
            # Если буфер превысил максимальный размер, удаляем самый старый чанк
            if len(window_buffer) > max_buffer_chunks:
                window_buffer.pop(0)
            
            # Объединяем все чанки в буфере в один файл для распознавания
            combined_file = "combined_chunk.wav"
            wf = wave.open(combined_file, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(RATE)
            
            # Записываем все фреймы из буфера
            for frames in window_buffer:
                wf.writeframes(b''.join(frames))
            wf.close()
            
            # Распознавание речи
            transcription = transcribe_chunk(model, combined_file)
            
            # Вывод в консоль
            print(f"\r[Чанк {chunk_counter}] {transcription}", end="", flush=True)
            
            # Если текст не пустой, добавляем его к общей транскрипции
            if transcription.strip():
                # Добавляем пробел только если уже есть текст
                if accumulated_transcription:
                    accumulated_transcription += " "
                accumulated_transcription += transcription
                
                # Проверяем, есть ли новый текст для записи в файл
                new_text = accumulated_transcription[len(last_saved_transcription):].strip()
                
                # Если есть новый текст - разбиваем на предложения и записываем в файл
                if new_text:
                    # Простое деление на предложения (по точке, восклицательному и вопросительному знакам)
                    sentences = []
                    current_sentence = ""
                    
                    for char in new_text:
                        current_sentence += char
                        if char in ['.', '!', '?'] and current_sentence.strip():
                            sentences.append(current_sentence.strip())
                            current_sentence = ""
                    
                    # Добавляем оставшийся текст как предложение, если он не пустой
                    if current_sentence.strip():
                        sentences.append(current_sentence.strip())
                    
                    # Записываем предложения в файл, каждое с новой строки
                    with open(output_file, "a", encoding="utf-8") as f:
                        for sentence in sentences:
                            f.write(f"{sentence}\n")
                    
                    # Обновляем последнюю сохраненную транскрибацию
                    last_saved_transcription = accumulated_transcription
                
                # Печатаем новую строку, чтобы показать накопленный текст
                print(f"\n[Текущая транскрипция] {accumulated_transcription}")
            
            # Удаляем временные файлы
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                if os.path.exists(combined_file):
                    os.remove(combined_file)
            except Exception as e:
                print(f"\nОшибка при удалении временных файлов: {e}")
    
    except KeyboardInterrupt:
        print("\nОстановка записи...")
    
    finally:
        # Остановка и закрытие потока
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        p.terminate()
        
        # Сохранение полной транскрипции в файл
        if accumulated_transcription:
            # Добавляем пометку о завершении записи
            with open(output_file, "a", encoding="utf-8") as f:
                f.write("\n=== Конец транскрибации ===\n")
            
            print(f"\nПолная транскрипция сохранена в {output_file}")
        else:
            print("\nНе было распознано ни одного текста.")

if __name__ == "__main__":
    main() 