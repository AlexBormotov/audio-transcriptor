"""
Модуль транскрипции аудио/видео файлов.

Использует WhisperLive (faster-whisper backend) для распознавания речи.
Поддерживает GPU (CUDA) с автоматическим переключением на CPU
при отсутствии CUDA или необходимых библиотек (cublas, cudnn).
"""

import torch
try:
    # Основной backend: WhisperLive (по запросу пользователя).
    from whisper_live.transcriber.transcriber_faster_whisper import (
        WhisperModel as BackendWhisperModel,
    )
    _backend_name = "whisper-live"
    _backend_import_error = None
except Exception as import_error:
    # Безопасный fallback: если WhisperLive не импортируется, используем faster-whisper напрямую.
    from faster_whisper import WhisperModel as BackendWhisperModel
    _backend_name = "faster-whisper"
    _backend_import_error = import_error


# Кэш загруженных моделей: {model_size: (WhisperModel, device)}
_model_cache = {}

# Актуальное устройство после инициализации (может измениться при fallback)
_active_device = None
_active_compute = None
_backend_warning_printed = False


def get_backend_info():
    """Возвращает backend транскрипции и причину fallback (если есть)."""
    if _backend_import_error is None:
        return _backend_name, None
    return _backend_name, str(_backend_import_error)


def get_device_info():
    """Определяет доступное устройство и тип вычислений.

    Проверяет наличие CUDA через torch. Фактическое устройство может
    измениться при загрузке модели, если CUDA-библиотеки недоступны.

    Returns:
        tuple: (device, compute_type) — например ("cuda", "float16") или ("cpu", "int8")
    """
    if _active_device is not None:
        return _active_device, _active_compute
    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "float32"


def _try_load_model(model_size, device, compute_type):
    """Пытается загрузить модель на указанном устройстве.

    Returns:
        WhisperModel: загруженная модель
    Raises:
        Exception: если загрузка не удалась
    """
    return BackendWhisperModel(model_size, device=device, compute_type=compute_type)


def get_model(model_size="base"):
    """Загружает и кэширует модель Whisper.

    При ошибке инициализации GPU (например, отсутствие cublas64_12.dll)
    автоматически переключается на CPU.

    Args:
        model_size: размер модели (tiny, base, small, medium, large-v3)

    Returns:
        WhisperModel: загруженная модель
    """
    global _active_device, _active_compute, _backend_warning_printed

    if model_size in _model_cache:
        return _model_cache[model_size]

    if _backend_import_error is not None and not _backend_warning_printed:
        print("[WARN] WhisperLive backend недоступен, используем faster-whisper.")
        print(f"[WARN] Причина: {_backend_import_error}")
        _backend_warning_printed = True

    device, compute_type = get_device_info()

    try:
        model = _try_load_model(model_size, device, compute_type)
        _active_device = device
        _active_compute = compute_type
    except Exception as e:
        if device == "cuda":
            print(f"[WARN] GPU недоступен: {e}")
            print("[WARN] Переключение на CPU (float32)...")
            device = "cpu"
            compute_type = "float32"
            model = _try_load_model(model_size, device, compute_type)
            _active_device = device
            _active_compute = compute_type
        else:
            raise

    _model_cache[model_size] = model
    return model


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

    kwargs = {
        "beam_size": 5,
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.3,
    }
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
