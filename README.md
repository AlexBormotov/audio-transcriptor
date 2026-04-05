# Транскрибатор аудио и видео

Windows-приложение для транскрипции аудио/видео файлов с помощью WhisperLive (backend: faster-whisper). Поддерживает GPU (CUDA) и автоматически переключается на CPU при недоступности.

## Возможности

- **Десктопное приложение** (PySide6) — drag & drop файлов, нативный Windows UI
- **Транскрипция файлов**: mp3, mp4, wav, flac, ogg, m4a, webm, mkv, avi
- **Два формата вывода**: сплошной текст или с таймкодами `[HH:MM:SS.mmm --> HH:MM:SS.mmm]`
- **Копирование и сохранение результата** в .txt файл
- **GPU ускорение** (CUDA) с автоматическим fallback на CPU
- **WhisperLive backend** по умолчанию (с fallback на faster-whisper)
- **Мультиязычность**: русский, английский и 10+ других языков
- **Веб-интерфейс** (Gradio) — альтернативный режим через браузер
- **Сборка установщика** — один .exe-файл для Windows x64

## Требования

- Python 3.10, 3.11 или 3.12
- ffmpeg (для конвертации видео)
- Видеокарта NVIDIA с CUDA (опционально, для ускорения)

## Установка (Windows)

1. Создайте виртуальное окружение:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

2. Установите PyTorch с поддержкой GPU:

```powershell
# GPU (CUDA 12.1) — рекомендуется при наличии NVIDIA GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Или CPU-only (если нет GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

3. Установите остальные зависимости:

```powershell
pip install -r requirements.txt
```

4. Установите ffmpeg (если не установлен):
   - Через winget: `winget install ffmpeg`
   - Или скачайте с https://ffmpeg.org/download.html
   - Убедитесь, что `ffmpeg` доступен в PATH

## Запуск

### Десктопное приложение (рекомендуется)

```powershell
python main.py
```

Откроется окно с drag & drop зоной для загрузки файлов.

### Веб-интерфейс (альтернативный)

```powershell
python app.py
```

Откроется в браузере по адресу http://localhost:7860

### Онлайн-распознавание с микрофона (legacy)

```powershell
python speech_recognition_online.py
```

## Сборка Windows-установщика

### Шаг 1: Сборка .exe через PyInstaller

```powershell
pip install pyinstaller
python build_exe.py
```

Результат: папка `dist/Transcriber/` с `Transcriber.exe` и всеми зависимостями.

### Шаг 2: Создание установщика через Inno Setup

1. Скачайте [Inno Setup](https://jrsoftware.org/isdl.php) и установите
2. Откройте `installer.iss` в Inno Setup Compiler
3. Нажмите **Build → Compile**
4. Готовый установщик: `installer_output/TranscriberSetup_x64.exe`

### Иконка приложения (опционально)

Поместите `icon.ico` (формат ICO, 256×256) в корень проекта перед сборкой.

## Архитектура

```
main.py              — точка входа десктопного приложения
main_window.py       — главное окно PySide6 (drag & drop, настройки, результат)
worker.py            — фоновый поток транскрипции (QThread)
transcriber.py       — ядро транскрипции (faster-whisper / WhisperLive)
constants.py         — общие константы (модели, языки, расширения)
app.py               — альтернативный веб-интерфейс (Gradio)
build_exe.py         — скрипт сборки .exe через PyInstaller
installer.iss        — конфигурация установщика Inno Setup
```

## Устранение проблем

### cublas64_12.dll not found

CUDA-библиотеки не найдены. Решения:

1. Установите CUDA Toolkit 12.x: https://developer.nvidia.com/cuda-downloads
2. Или приложение автоматически переключится на CPU

### Конфликты зависимостей при pip install

```powershell
# Установите PyTorch ОТДЕЛЬНО (перед requirements.txt)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

После установки обновите faster-whisper:

```powershell
pip install --upgrade --no-deps faster-whisper==1.2.0
```

### ffmpeg не найден

- Windows: `winget install ffmpeg`
- Linux: `sudo apt install ffmpeg`

### Размер сборки слишком большой

PyTorch + CUDA занимает ~2 ГБ. Для уменьшения:
- Используйте CPU-only PyTorch (меньше на ~1 ГБ)
- Используйте UPX-сжатие: добавьте `--upx-dir=path/to/upx` в build_exe.py

---

**Приложение автоматически определяет GPU и переключается на CPU при недоступности CUDA.**
