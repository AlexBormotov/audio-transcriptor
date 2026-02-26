"""
Фоновый поток для транскрипции файлов.

Выполняет конвертацию и распознавание в отдельном QThread,
чтобы не блокировать UI.
"""

import os
import subprocess
import tempfile

from PySide6.QtCore import QThread, Signal

from transcriber import transcribe_file


class TranscribeWorker(QThread):
    """Выполняет транскрипцию файла в фоновом потоке."""

    # Сигналы для связи с главным окном
    finished = Signal(str, dict)   # (текст, lang_info)
    error = Signal(str)            # сообщение об ошибке
    progress = Signal(str)         # статус-сообщение

    def __init__(self, file_path, model_size, language, output_format):
        super().__init__()
        self.file_path = file_path
        self.model_size = model_size
        self.language = language
        self.output_format = output_format

    def run(self):
        """Основной цикл: конвертация -> транскрипция -> результат."""
        wav_path = None
        try:
            self.progress.emit("Конвертация аудио в WAV...")
            wav_path = self._convert_to_wav(self.file_path)

            self.progress.emit(f"Загрузка модели {self.model_size}...")

            lang = self.language if self.language != "auto" else None
            self.progress.emit("Транскрипция (может занять несколько минут)...")
            text, lang_info = transcribe_file(
                wav_path, self.model_size, lang, self.output_format
            )

            if not text.strip():
                text = "(Речь не распознана. Попробуйте другой файл или модель.)"

            self.finished.emit(text, lang_info)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            # Чистим временный WAV-файл
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

    @staticmethod
    def _convert_to_wav(input_path):
        """Конвертирует аудио/видео в WAV 16kHz mono через ffmpeg."""
        wav_path = tempfile.mktemp(suffix=".wav")
        cmd = [
            "ffmpeg", "-i", input_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "-y", wav_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg ошибка: {result.stderr[:500]}")
        return wav_path
