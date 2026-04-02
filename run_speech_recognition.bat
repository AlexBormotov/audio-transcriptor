@echo off
rem Без аргументов: микрофон (speech_recognition_online.py, нужна CUDA).
rem С аргументом: транскрибация файла — transcriber.py (как в GUI).
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  python speech_recognition_online.py
) else (
  python transcriber.py --model base %*
)
pause
 