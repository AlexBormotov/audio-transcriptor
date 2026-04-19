"""
Запись звука с микрофона.

Содержит фоновый поток для захвата аудио через sounddevice
и утилиту для получения списка входных устройств.
"""

import struct
import tempfile
import threading
import wave

from PySide6.QtCore import QThread, Signal

from constants import DEFAULT_SAMPLE_RATE

try:
    import sounddevice as sd
    _sd_available = True
    _sd_import_error = None
except Exception as e:
    sd = None
    _sd_available = False
    _sd_import_error = e


def is_available():
    """Проверяет, доступна ли библиотека sounddevice."""
    return _sd_available


def get_import_error():
    """Возвращает ошибку импорта sounddevice (или None)."""
    return _sd_import_error


def get_input_devices():
    """Возвращает список доступных входных устройств.

    Returns:
        list[dict]: список словарей {"index": int, "name": str}
    """
    if not _sd_available:
        return []
    devices = []
    try:
        for info in sd.query_devices():
            if info["max_input_channels"] > 0:
                devices.append({
                    "index": info["index"],
                    "name": info["name"],
                })
    except Exception:
        pass
    return devices


class MicRecorderWorker(QThread):
    """Записывает звук с микрофона в фоновом потоке.

    Signals:
        finished(str): путь к временному WAV-файлу после остановки записи
        error(str): сообщение об ошибке
        level(float): текущий уровень громкости 0.0–1.0 (обновляется ~10 раз/сек)
    """

    finished = Signal(str)
    error = Signal(str)
    level = Signal(float)

    def __init__(self, device_index=None, sample_rate=DEFAULT_SAMPLE_RATE):
        """
        Args:
            device_index: индекс устройства sounddevice (None = системное по умолчанию)
            sample_rate: частота дискретизации (по умолчанию 16000)
        """
        super().__init__()
        self.device_index = device_index
        self.sample_rate = sample_rate
        self._stop_event = threading.Event()

    def stop(self):
        """Запрашивает остановку записи."""
        self._stop_event.set()

    def run(self):
        """Основной цикл: запись → сохранение WAV → emit finished."""
        if not _sd_available:
            self.error.emit("Библиотека sounddevice не установлена.")
            return

        frames = []
        block_size = self.sample_rate // 10  # ~100 мс на блок → 10 обновлений/сек

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=block_size,
                device=self.device_index,
            ) as stream:
                while not self._stop_event.is_set():
                    data, overflowed = stream.read(block_size)
                    frames.append(data.tobytes())

                    # Вычисляем RMS-уровень и нормируем к 0–1
                    samples = struct.unpack(f"<{len(data)//2}h", data.tobytes())
                    if samples:
                        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
                        # Нормируем: 32767 — максимум int16
                        normalized = min(rms / 32767.0 * 5.0, 1.0)
                        self.level.emit(normalized)

        except Exception as e:
            self.error.emit(f"Ошибка записи с микрофона: {e}")
            return

        if not frames:
            self.error.emit("Запись пуста — не удалось захватить звук.")
            return

        # Сохраняем во временный WAV
        try:
            wav_path = tempfile.mktemp(suffix=".wav")
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16 = 2 байта
                wf.setframerate(self.sample_rate)
                wf.writeframes(b"".join(frames))
            self.finished.emit(wav_path)
        except Exception as e:
            self.error.emit(f"Не удалось сохранить запись: {e}")
