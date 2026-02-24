# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a single-file Python CLI application for real-time speech-to-text transcription from a microphone using the Whisper model (`faster-whisper`). Main entry point: `speech_recognition_online.py`.

### Hardware limitations in Cloud VM

- **No NVIDIA GPU / CUDA**: The app requires `torch.cuda.is_available() == True` and will `sys.exit(1)` otherwise. In the cloud VM, only CPU-mode PyTorch is available. To test transcription logic on CPU, load the model with `device='cpu', compute_type='int8'`.
- **No microphone**: PyAudio cannot open an input stream without a sound device. ALSA warnings are expected and harmless.

### Python environment

- Python 3.11 is installed via `deadsnakes` PPA (`python3.11`). The system default `python3` is 3.12, which is **incompatible** with `numpy==1.24.3`.
- Virtual environment lives at `/workspace/venv` and uses Python 3.11.
- Activate with `source /workspace/venv/bin/activate`.
- PyTorch is installed as CPU-only (`--index-url https://download.pytorch.org/whl/cpu`).

### Running / testing

- `python speech_recognition_online.py` — will exit immediately with "CUDA не обнаружена" error (expected in Cloud VM).
- `python test_torch_cuda.py` — will show CUDA status; in Cloud VM `torch.cuda.is_available()` returns `False`.
- To test Whisper transcription on CPU: load `WhisperModel('base', device='cpu', compute_type='int8')` and call `model.transcribe(wav_path)`.

### System dependencies

- `portaudio19-dev` and `libasound2-dev` are required for PyAudio to compile.

### No linting / testing / build system

This project has no linter, no test suite, and no build system configured. It is a single Python script with `requirements.txt`.
