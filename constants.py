"""
Общие константы приложения-транскрибатора.

Используется как в десктопном (main_window.py), так и в веб-интерфейсе (app.py).
"""

# Доступные размеры моделей Whisper.
MODEL_SIZES = ["large-v3", "medium", "small", "base", "tiny"]

# Поддерживаемые языки: отображаемое имя -> код для Whisper
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

# Поддерживаемые расширения аудио/видео файлов
SUPPORTED_EXTENSIONS = {
    ".mp3", ".mp4", ".wav", ".flac", ".ogg",
    ".m4a", ".wma", ".aac", ".webm", ".mkv", ".avi",
}

# Контейнеры с видеодорожкой: для превью показываем QVideoWidget (остальное — аудио-панель)
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi"}

# Частота дискретизации для записи с микрофона (совпадает с ожиданиями Whisper)
DEFAULT_SAMPLE_RATE = 16000

# Версия приложения
APP_VERSION = "1.0.0"
APP_NAME = "Транскрибатор"
APP_NAME_EN = "Transcriber"
