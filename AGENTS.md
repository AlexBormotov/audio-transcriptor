# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a single-script Python application for real-time speech-to-text transcription from microphone using Whisper (`faster-whisper`). The main entry point is `speech_recognition_online.py`.

### Key constraints in Cloud VM

- **No CUDA GPU**: The app hard-requires `torch.cuda.is_available()` and exits if CUDA is not found. In the cloud VM, only CPU is available. To test model loading/transcription, use `device='cpu'` and `compute_type='int8'` instead of `cuda`/`float16`.
- **No microphone**: The app requires a physical/virtual audio input device via PyAudio. The cloud VM has no audio device. Functions that don't touch the microphone (e.g., `format_text_with_newlines`, `transcribe_chunk`, `recognize_and_save`) can be tested independently.
- **Python 3.12**: The VM has Python 3.12, while `requirements.txt` pins versions for Python 3.10/3.11. Dependencies are installed without strict version pins to maintain compatibility.

### Architecture

- `transcriber.py` — ядро транскрипции, использует whisper-live/faster-whisper. Автоматический GPU/CPU fallback.
- `app.py` — Gradio веб-интерфейс (запуск: `python app.py`, порт 7860).
- `speech_recognition_online.py` — оригинальный скрипт для микрофона (требует CUDA + микрофон).

### Running the development environment

1. Activate the virtual environment: `source /workspace/venv/bin/activate`
2. Start the web UI: `python app.py` (opens at http://localhost:7860)
3. All dependencies (including PyTorch CPU) are pre-installed in the venv.

### Linting

```bash
source /workspace/venv/bin/activate
flake8 transcriber.py app.py --max-line-length=120
```

### Testing transcription (CPU-only)

```python
from transcriber import transcribe_file
text, info = transcribe_file('combined_chunk.wav', 'base', None, 'plain')
print(text)
```

### Gotchas

- PyAudio requires the `portaudio19-dev` and `python3-dev` system packages to compile.
- `speech_recognition_online.py` cannot run in the cloud VM (needs CUDA + microphone). Use `app.py` instead.
- `gr.File(type="filepath")` in Gradio 6.x may not render text in output Textbox. Use `gr.Audio(type="filepath")` instead for file upload.
- Video files (mp4, mkv, avi) are preprocessed through ffmpeg to extract audio before transcription.
- The `venv/` directory is in `.gitignore` and should not be committed.
