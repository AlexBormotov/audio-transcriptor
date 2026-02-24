"""
Модуль транскрипции аудио/видео файлов.

Использует whisper-live (faster-whisper backend) для распознавания речи.
Поддерживает GPU (CUDA) с автоматическим переключением на CPU.
"""

import torch
from faster_whisper import WhisperModel


# Кэш загруженных моделей: {model_size: WhisperModel}
_model_cache = {}


def get_device_info():
    """Определяет доступное устройство и тип вычислений.

    Returns:
        tuple: (device, compute_type) — например ("cuda", "float16") или ("cpu", "int8")
    """
    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "int8"


def get_model(model_size="base"):
    """Загружает и кэширует модель Whisper.

    Args:
        model_size: размер модели (tiny, base, small, medium, large-v3)

    Returns:
        WhisperModel: загруженная модель
    """
    if model_size not in _model_cache:
        device, compute_type = get_device_info()
        _model_cache[model_size] = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )
    return _model_cache[model_size]


def format_timestamp(seconds):
    """Форматирует секунды в HH:MM:SS.mmm.

    Args:
        seconds: время в секундах (float)

    Returns:
        str: отформатированная строка, например "00:01:23.456"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def transcribe_file(file_path, model_size="base", language=None,
                    output_format="plain"):
    """Транскрибирует аудио/видео файл.

    Поддерживает форматы: mp3, mp4, wav, flac, ogg, m4a, webm, mkv, avi и др.
    Формат файла определяется автоматически через библиотеку av (PyAV).

    Args:
        file_path: путь к аудио/видео файлу
        model_size: размер модели Whisper (tiny, base, small, medium, large-v3)
        language: код языка ("ru", "en" и т.д.) или None для автоопределения
        output_format: "plain" — сплошной текст, "timestamps" — с таймкодами

    Returns:
        tuple: (текст транскрипции, dict с информацией о языке)
    """
    model = get_model(model_size)

    kwargs = {"beam_size": 5}
    if language and language != "auto":
        kwargs["language"] = language

    segments, info = model.transcribe(file_path, **kwargs)

    if output_format == "timestamps":
        lines = []
        for segment in segments:
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            lines.append(f"[{start} --> {end}] {segment.text.strip()}")
        result = "\n".join(lines)
    else:
        texts = []
        for segment in segments:
            texts.append(segment.text.strip())
        result = " ".join(texts)

    lang_info = {
        "language": info.language,
        "probability": info.language_probability,
    }
    return result, lang_info
