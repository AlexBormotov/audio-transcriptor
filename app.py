"""
Веб-интерфейс транскрибатора аудио/видео файлов.

Запуск: python app.py
Откроется в браузере по адресу http://localhost:7860
"""

import os
import subprocess
import tempfile

import gradio as gr

from transcriber import transcribe_file, get_backend_info, get_device_info

# Доступные размеры моделей Whisper
MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]

# Поддерживаемые языки: отображаемое имя -> код
LANGUAGES = {
    "Автоопределение": "auto",
    "Русский": "ru",
    "English": "en",
    "Deutsch": "de",
    "Français": "fr",
    "Español": "es",
    "中文": "zh",
    "日本語": "ja",
    "한국어": "ko",
    "Italiano": "it",
    "Português": "pt",
    "Türkçe": "tr",
    "العربية": "ar",
    "हिन्दी": "hi",
}


def convert_to_wav(input_path):
    """Конвертирует аудио/видео файл в WAV 16kHz mono через ffmpeg.

    Returns:
        str: путь к сконвертированному WAV-файлу
    """
    wav_path = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        "-y", wav_path,
    ]
    # На Windows вывод ffmpeg может содержать байты вне cp1252.
    # Явно читаем как UTF-8 и не падаем на "битых" символах.
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg ошибка: {result.stderr[:300]}")
    return wav_path


def process_file(file_path, model_size, language_name, output_format_name):
    """Обрабатывает загруженный файл и возвращает транскрипцию.

    Args:
        file_path: путь к загруженному файлу
        model_size: размер модели Whisper
        language_name: отображаемое имя языка
        output_format_name: "Сплошной текст" или "С таймкодами"

    Returns:
        tuple: (текст, путь к файлу для скачивания, информация)
    """
    if file_path is None:
        raise gr.Error("Пожалуйста, загрузите файл для транскрипции.")

    if not os.path.exists(file_path):
        raise gr.Error(f"Файл не найден: {file_path}")

    # Конвертируем в WAV 16kHz mono для стабильной работы Whisper
    wav_path = convert_to_wav(file_path)

    lang_code = LANGUAGES.get(language_name, "auto")
    fmt = "timestamps" if output_format_name == "С таймкодами" else "plain"

    try:
        text, lang_info = transcribe_file(
            wav_path, model_size, lang_code, fmt
        )
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

    if not text.strip():
        text = "(Речь не распознана. Попробуйте другой файл или модель.)"
        gr.Warning("Транскрипция не дала результатов. Попробуйте повторить.")

    # Сохраняем результат во временный файл для скачивания
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    suffix = "_timestamps" if fmt == "timestamps" else "_plain"
    output_filename = f"{base_name}{suffix}.txt"
    output_path = os.path.join(tempfile.gettempdir(), output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    detected = lang_info["language"]
    prob = lang_info["probability"]
    # get_device_info возвращает актуальное устройство после fallback
    actual_device, actual_compute = get_device_info()
    backend_name, backend_error = get_backend_info()
    backend_label = backend_name
    if backend_error:
        backend_label = f"{backend_name} (fallback)"
    info_text = (
        f"Язык: {detected} ({prob:.0%}) | "
        f"Устройство: {actual_device.upper()} ({actual_compute}) | "
        f"Backend: {backend_label} | "
        f"Модель: {model_size}"
    )

    return text, output_path, info_text


def build_ui():
    """Создаёт и возвращает Gradio-интерфейс."""
    device, compute_type = get_device_info()
    backend_name, _ = get_backend_info()
    if device == "cuda":
        device_badge = "🟢 GPU (CUDA)"
    else:
        device_badge = "🟡 CPU"

    with gr.Blocks(title="Транскрибатор") as demo:
        gr.Markdown(
            f"# 🎙️ Транскрибатор аудио и видео\n"
            f"Загрузите файл (mp3, mp4, wav, flac, ogg, m4a, webm, mkv, avi) "
            f"и получите текстовую транскрипцию.\n\n"
            f"**Устройство:** {device_badge} ({compute_type})  \n"
            f"**Backend:** {backend_name}"
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Загрузите аудио/видео файл",
                    file_types=[
                        ".mp3", ".mp4", ".wav", ".flac", ".ogg",
                        ".m4a", ".wma", ".aac", ".webm", ".mkv", ".avi",
                    ],
                    type="filepath",
                )
                model_size = gr.Dropdown(
                    choices=MODEL_SIZES,
                    value="base",
                    label="Размер модели",
                    info="base — баланс скорости и качества",
                )
                language = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="Автоопределение",
                    label="Язык",
                    info="Автоопределение обычно работает хорошо",
                )
                output_format = gr.Radio(
                    choices=["Сплошной текст", "С таймкодами"],
                    value="Сплошной текст",
                    label="Формат вывода",
                )
                transcribe_btn = gr.Button(
                    "▶ Транскрибировать",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=2):
                info_label = gr.Textbox(
                    label="Информация",
                    interactive=False,
                    max_lines=1,
                )
                output_text = gr.Textbox(
                    label="Результат транскрипции",
                    lines=18,
                    max_lines=50,
                    interactive=False,
                )
                download_file = gr.File(
                    label="Скачать результат (.txt)",
                )

        transcribe_btn.click(
            fn=process_file,
            inputs=[file_input, model_size, language, output_format],
            outputs=[output_text, download_file, info_label],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        show_error=True,
    )
