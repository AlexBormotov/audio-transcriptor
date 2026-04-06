"""
Простое превью выбранного файла: видео (QVideoWidget) или аудио (заглушка + ползунок).
Кнопка «Открыть файл» снизу; выбор файла дублирует логику главного окна.
"""

import os

from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QSlider,
    QStackedWidget,
    QSizePolicy,
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget

from constants import SUPPORTED_EXTENSIONS, VIDEO_EXTENSIONS


def _fmt_ms(ms: int) -> str:
    """Форматирует миллисекунды как m:ss для подписи времени."""
    if ms < 0:
        ms = 0
    s = ms // 1000
    m, s = s // 60, s % 60
    return f"{m}:{s:02d}"


class MediaPreviewWidget(QWidget):
    """
    Воспроизведение локального файла через QMediaPlayer.
    Сигнал file_selected — когда пользователь выбрал другой файл кнопкой «Открыть».
    """

    file_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._player = QMediaPlayer(self)
        self._audio_out = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_out)

        self._preview_stack = QStackedWidget()
        self._video = QVideoWidget()
        self._video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._video.setMinimumHeight(180)
        self._audio_panel = QWidget()
        ap_layout = QVBoxLayout(self._audio_panel)
        self._audio_label = QLabel("🔊 Аудио")
        self._audio_label.setAlignment(Qt.AlignCenter)
        self._audio_label.setMinimumHeight(180)
        self._audio_label.setWordWrap(True)
        ap_layout.addWidget(self._audio_label)

        self._preview_stack.addWidget(self._video)
        self._preview_stack.addWidget(self._audio_panel)

        self._time_label = QLabel("0:00 / 0:00")
        self._time_label.setMinimumWidth(120)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.setEnabled(False)

        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedWidth(44)
        self._play_btn.clicked.connect(self._toggle_play)

        self._open_btn = QPushButton("📂 Открыть файл…")
        self._open_btn.clicked.connect(self._pick_file)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #b00020;")
        self._error_label.setWordWrap(True)

        controls = QHBoxLayout()
        controls.addWidget(self._play_btn)
        controls.addWidget(self._time_label)
        controls.addWidget(self._slider, stretch=1)

        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.addWidget(self._preview_stack, stretch=1)
        root.addLayout(controls)
        root.addWidget(self._open_btn)
        root.addWidget(self._error_label)

        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.playbackStateChanged.connect(self._on_state_changed)
        self._player.errorOccurred.connect(self._on_player_error)
        self._slider.sliderMoved.connect(self._player.setPosition)

    def load(self, path: str) -> None:
        """Подставляет файл в плеер: видео или только аудио по расширению."""
        self._error_label.clear()
        self._player.stop()
        ext = os.path.splitext(path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            self._player.setVideoOutput(self._video)
            self._preview_stack.setCurrentIndex(0)
        else:
            self._player.setVideoOutput(None)
            self._preview_stack.setCurrentIndex(1)
            self._audio_label.setText(f"🔊 Аудио\n{os.path.basename(path)}")
        self._player.setSource(QUrl.fromLocalFile(path))
        self._slider.setValue(0)
        self._time_label.setText("0:00 / 0:00")

    def set_interactive(self, enabled: bool) -> None:
        """Во время транскрипции отключаем управление (и ставим на паузу)."""
        if not enabled:
            self._player.pause()
        self._play_btn.setEnabled(enabled)
        self._open_btn.setEnabled(enabled)
        self._slider.setEnabled(enabled and self._player.duration() > 0)

    def shutdown(self) -> None:
        """Освободить плеер при закрытии окна."""
        self._player.stop()

    def _pick_file(self) -> None:
        ext_filter = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл", "", f"Аудио/Видео ({ext_filter})"
        )
        if path:
            self.file_selected.emit(path)

    def _toggle_play(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_state_changed(self, _state) -> None:
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._play_btn.setText("⏸")
        else:
            self._play_btn.setText("▶")

    def _on_duration_changed(self, duration: int) -> None:
        d = max(int(duration), 0)
        self._slider.setMaximum(max(d, 1))
        self._slider.setEnabled(d > 0 and self._open_btn.isEnabled())
        self._update_time_label()

    def _on_position_changed(self, pos: int) -> None:
        if self._slider.isSliderDown():
            return
        self._slider.blockSignals(True)
        self._slider.setValue(int(pos))
        self._slider.blockSignals(False)
        self._update_time_label()

    def _update_time_label(self) -> None:
        self._time_label.setText(
            f"{_fmt_ms(self._player.position())} / {_fmt_ms(self._player.duration())}"
        )

    def _on_player_error(self, _error, error_string: str) -> None:
        self._error_label.setText(
            error_string or "Не удалось воспроизвести файл (кодек или формат)."
        )
