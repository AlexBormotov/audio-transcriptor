"""
Фоновый поток для транскрипции файлов.

Выполняет конвертацию и распознавание в отдельном QThread,
чтобы не блокировать UI.
"""

import os

from PySide6.QtCore import QThread, Signal

from transcriber import transcribe_file, media_to_wav_16k_mono


class TranscribeWorker(QThread):
    """Выполняет транскрипцию файла в фоновом потоке."""

    # Сигналы для связи с главным окном
    finished = Signal(str, dict)   # (текст, lang_info)
    error = Signal(str)            # сообщение об ошибке
    progress = Signal(str)         # статус-сообщение

    def __init__(self, file_path, model_size, language, output_format,
                 skip_convert=False):
        super().__init__()
        self.file_path = file_path
        self.model_size = model_size
        self.language = language
        self.output_format = output_format
        self.skip_convert = skip_convert

    def run(self):
        """Основной цикл: конвертация -> транскрипция -> результат."""
        wav_path = None
        try:
            if self.skip_convert:
                # Файл уже в формате WAV 16 kHz моно (например, запись с микрофона)
                wav_path = self.file_path
            else:
                self.progress.emit("Конвертация аудио в WAV...")
                # Длинные ролики: без жёсткого таймаута ffmpeg (раньше 300 с могло обрывать конвертацию).
                wav_path = media_to_wav_16k_mono(self.file_path, timeout=None)

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
