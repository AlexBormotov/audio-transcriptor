"""
Модуль транскрипции аудио/видео файлов.

Использует WhisperLive (faster-whisper backend) для распознавания речи.
Поддерживает GPU (CUDA) с автоматическим переключением на CPU
при отсутствии CUDA или необходимых библиотек (cublas, cudnn).
"""

import os
import subprocess
import sys
import tempfile

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


# Кэш загруженных моделей: {(model_size, force_cpu): WhisperModel}
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


def _should_retry_on_cpu(err):
    """Ошибки CUDA/cuBLAS, которые часто всплывают уже на этапе transcribe(), не при load()."""
    msg = str(err).lower()
    if "cublas" in msg or "cudnn" in msg:
        return True
    if "cuda" in msg and ("dll" in msg or "not found" in msg or "cannot be loaded" in msg):
        return True
    return False


def _try_load_model(model_size, device, compute_type):
    """Пытается загрузить модель на указанном устройстве.

    Returns:
        WhisperModel: загруженная модель
    Raises:
        Exception: если загрузка не удалась
    """
    return BackendWhisperModel(model_size, device=device, compute_type=compute_type)


def get_model(model_size="base", force_cpu=False):
    """Загружает и кэширует модель Whisper.

    При ошибке инициализации GPU (например, отсутствие cublas64_12.dll)
    автоматически переключается на CPU.

    Args:
        model_size: размер модели (tiny, base, small, medium, large-v3)
        force_cpu: принудительно CPU (для CLI и повтора после ошибки CUDA на transcribe)

    Returns:
        WhisperModel: загруженная модель
    """
    global _active_device, _active_compute, _backend_warning_printed

    cache_key = (model_size, force_cpu)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if _backend_import_error is not None and not _backend_warning_printed:
        print("[WARN] WhisperLive backend недоступен, используем faster-whisper.")
        print(f"[WARN] Причина: {_backend_import_error}")
        _backend_warning_printed = True

    if force_cpu:
        device, compute_type = "cpu", "float32"
        model = _try_load_model(model_size, device, compute_type)
        _active_device = device
        _active_compute = compute_type
        _model_cache[cache_key] = model
        return model

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

    _model_cache[cache_key] = model
    return model


def media_to_wav_16k_mono(input_path, timeout=None):
    """Извлекает аудио в моно WAV 16 kHz через ffmpeg (как в GUI и Gradio).

    Тот же путь, что и в worker/app: стабильная подача в Whisper для mp4/mkv и т.д.
    Вызывающий код обязан удалить временный файл после использования.

    Args:
        input_path: исходный аудио/видео файл
        timeout: лимит секунд для ffmpeg (None = без лимита, для длинных роликов)

    Returns:
        str: путь к временному .wav
    """
    wav_path = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        "-y", wav_path,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.returncode != 0:
        try:
            os.remove(wav_path)
        except OSError:
            pass
        raise RuntimeError(f"ffmpeg ошибка: {result.stderr[:500]}")
    return wav_path


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
                    output_format="plain", force_cpu=False):
    """Транскрибирует аудио/видео файл.

    Поддерживает форматы: mp3, mp4, wav, flac, ogg, m4a, webm, mkv, avi и др.
    Формат файла определяется автоматически через библиотеку av (PyAV).

    Args:
        file_path: путь к аудио/видео файлу
        model_size: размер модели Whisper (tiny, base, small, medium, large-v3)
        language: код языка ("ru", "en" и т.д.) или None для автоопределения
        output_format: "plain" — сплошной текст, "timestamps" — с таймкодами
        force_cpu: не использовать GPU (или повтор после сбоя CUDA на transcribe)

    Returns:
        tuple: (текст транскрипции, dict с информацией о языке)
    """
    global _active_device, _active_compute

    model = get_model(model_size, force_cpu=force_cpu)

    kwargs = {
        "beam_size": 5,
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.3,
    }
    if language and language != "auto":
        kwargs["language"] = language

    try:
        segments, info = model.transcribe(file_path, **kwargs)
    except Exception as e:
        # faster-whisper иногда падает на transcribe() из‑за отсутствия cuBLAS при «живом» torch.cuda
        if not force_cpu and _should_retry_on_cpu(e):
            print(
                "[WARN] Ошибка GPU при распознавании; повтор на CPU…",
                file=sys.stderr,
            )
            _model_cache.pop((model_size, False), None)
            _active_device = None
            _active_compute = None
            return transcribe_file(
                file_path, model_size, language, output_format, force_cpu=True,
            )
        raise

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


if __name__ == "__main__":
    # Запуск из консоли: транскрибация файла без микрофона (в отличие от speech_recognition_online.py).
    import argparse

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description="Транскрибация аудио/видео через Whisper (тот же стек, что GUI/Gradio).",
    )
    parser.add_argument("file", help="Путь к медиафайлу")
    parser.add_argument("--model", default="base", help="Размер модели (tiny, base, ...)")
    parser.add_argument(
        "--language",
        default=None,
        help="Код языка Whisper (ru, en, …); не указывать — автоопределение",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Вывод с таймкодами по сегментам",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Путь к .txt (по умолчанию: <имя_файла>_transcript.txt рядом с видео)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Только CPU (если CUDA/cuBLAS на машине сломаны)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"Файл не найден: {args.file}", file=sys.stderr)
        sys.exit(1)

    fmt = "timestamps" if args.timestamps else "plain"
    lang = args.language if args.language else None

    wav_path = None
    try:
        print("[INFO] ffmpeg: извлечение аудио в WAV 16 kHz…")
        wav_path = media_to_wav_16k_mono(args.file)
        print("[INFO] Транскрипция…")
        text, lang_info = transcribe_file(
            wav_path, args.model, lang, fmt, force_cpu=args.cpu,
        )
    except Exception as err:
        print(f"[ОШИБКА] {err}", file=sys.stderr)
        sys.exit(1)
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass

    out_path = args.output
    if not out_path:
        root, _ = os.path.splitext(os.path.abspath(args.file))
        out_path = root + "_transcript.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")

    prob = lang_info.get("probability")
    prob_s = f"{prob:.2f}" if prob is not None else "n/a"
    print(f"[OK] Язык: {lang_info.get('language')} (вероятность {prob_s})")
    print(f"[OK] Текст сохранён: {out_path}")
