# Транскрибатор аудио и видео

Веб-приложение для транскрипции аудио/видео файлов с помощью WhisperLive (backend: faster-whisper). Поддерживает GPU (CUDA) и автоматически переключается на CPU при недоступности.

## Возможности

- **Транскрипция файлов**: загрузите mp3, mp4, wav, flac, ogg, m4a, webm, mkv, avi
- **Два формата вывода**: сплошной текст или с таймкодами `[HH:MM:SS.mmm --> HH:MM:SS.mmm]`
- **Скачивание результата** в .txt файл
- **GPU ускорение** (CUDA) с автоматическим fallback на CPU
- **WhisperLive backend** по умолчанию (с fallback на faster-whisper при проблемах импорта)
- **Мультиязычность**: русский, английский и 10+ других языков
- **Веб-интерфейс** (Gradio) — работает в браузере на любой ОС
- **Онлайн-распознавание с микрофона** (legacy режим, `speech_recognition_online.py`)

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
   - Скачайте с https://ffmpeg.org/download.html
   - Или через winget: `winget install ffmpeg`
   - Убедитесь, что `ffmpeg` доступен в PATH

## Запуск

### Веб-интерфейс (транскрипция файлов)

```powershell
python app.py
```

Откроется в браузере по адресу http://localhost:7860

### Онлайн-распознавание с микрофона (legacy)

```powershell
python speech_recognition_online.py
```

## Устранение проблем

### cublas64_12.dll not found

Эта ошибка означает, что CUDA-библиотеки не найдены. Решения:

1. **Установите CUDA Toolkit 12.x или 13.x** с сайта NVIDIA: https://developer.nvidia.com/cuda-downloads  
   (для PyTorch обычно используются колёса `cu121`; важна совместимость драйвера NVIDIA)
2. **Или**: приложение автоматически переключится на CPU — просто перезапустите `python app.py`

### Конфликты зависимостей при pip install

Если `pip install -r requirements.txt` даёт ошибки конфликтов:

```powershell
# Установите PyTorch ОТДЕЛЬНО (перед requirements.txt)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Затем остальные зависимости
pip install -r requirements.txt
```

`requirements.txt` ставит `WhisperLive` из GitHub `main` (это важно, потому что PyPI-релиз `0.7.1` содержит старый backend-файл).

После установки зависимостей обновите faster-whisper до версии из upstream `requirements/server.txt`:

```powershell
pip install --upgrade --no-deps faster-whisper==1.2.0
```

Почему отдельным шагом: metadata текущего пакета `whisper-live` всё ещё декларирует `faster-whisper==1.1.0`, поэтому прямой одновременный пин в `requirements.txt` вызывает конфликт резолвера `pip`.

### PyTorch не видит CUDA

- Убедитесь, что установлен PyTorch с CUDA (не CPU-версия)
- Проверьте: `python -c "import torch; print(torch.cuda.is_available())"`
- Используйте Python 3.10/3.11/3.12

### ffmpeg не найден

- Windows: `winget install ffmpeg` или скачайте с https://ffmpeg.org
- Linux: `sudo apt install ffmpeg`

---

**Приложение автоматически определяет GPU и переключается на CPU при недоступности CUDA.**
