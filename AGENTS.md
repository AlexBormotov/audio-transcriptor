# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a single-script Python application for real-time speech-to-text transcription from microphone using Whisper (`faster-whisper`). The main entry point is `speech_recognition_online.py`.

### Key constraints in Cloud VM

- **No CUDA GPU**: The app hard-requires `torch.cuda.is_available()` and exits if CUDA is not found. In the cloud VM, only CPU is available. To test model loading/transcription, use `device='cpu'` and `compute_type='int8'` instead of `cuda`/`float16`.
- **No microphone**: The app requires a physical/virtual audio input device via PyAudio. The cloud VM has no audio device. Functions that don't touch the microphone (e.g., `format_text_with_newlines`, `transcribe_chunk`, `recognize_and_save`) can be tested independently.
- **Python 3.12**: The VM has Python 3.12, while `requirements.txt` pins versions for Python 3.10/3.11. Dependencies are installed without strict version pins to maintain compatibility.

### Running the development environment

1. Activate the virtual environment: `source /workspace/venv/bin/activate`
2. All dependencies (including PyTorch CPU) are pre-installed in the venv.

### Linting

```bash
source /workspace/venv/bin/activate
flake8 speech_recognition_online.py --max-line-length=120
pylint speech_recognition_online.py --disable=C0301 --max-line-length=120
```

### Testing transcription (CPU-only)

To test Whisper transcription without GPU/microphone, use the included `combined_chunk.wav` sample:

```python
from faster_whisper import WhisperModel
model = WhisperModel('base', device='cpu', compute_type='int8')
segments, info = model.transcribe('combined_chunk.wav', beam_size=5)
text = ' '.join([s.text for s in segments])
print(text)
```

### Gotchas

- PyAudio requires the `portaudio19-dev` and `python3-dev` system packages to compile.
- The `main()` function cannot run in the cloud VM due to CUDA and microphone requirements. Test individual functions instead.
- The `venv/` directory is in `.gitignore` and should not be committed.
