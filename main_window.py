"""
Главное окно десктопного приложения-транскрибатора.

Содержит зону drag & drop, настройки модели/языка/формата,
область результата и кнопки копирования/сохранения.
Поддерживает тёмную/светлую тему, масштаб шрифта и сворачивание в трей.
"""

import os
import time

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QRadioButton, QTextEdit,
    QLabel, QFileDialog, QProgressBar,
    QButtonGroup, QGroupBox, QMessageBox, QApplication,
    QStackedWidget, QSystemTrayIcon, QMenu,
)
from PySide6.QtCore import Qt, Signal, QTimer, QSettings
from PySide6.QtGui import (
    QDragEnterEvent, QDropEvent, QCloseEvent,
    QAction, QIcon, QPixmap, QPainter, QColor, QPalette, QFont,
)

from constants import MODEL_SIZES, LANGUAGES, SUPPORTED_EXTENSIONS, APP_NAME
from media_preview import MediaPreviewWidget
from mic_recorder import MicRecorderWorker, get_input_devices, is_available as mic_available
from worker import TranscribeWorker
from transcriber import get_device_info, get_backend_info

# Минимальный и максимальный размер шрифта для масштабирования
_FONT_SIZE_MIN = 8
_FONT_SIZE_MAX = 32
_FONT_SIZE_DEFAULT = 13


def _make_tray_icon() -> QIcon:
    """Создаёт простую иконку для системного трея (синий круг с буквой Т)."""
    px = QPixmap(64, 64)
    px.fill(Qt.transparent)
    painter = QPainter(px)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor("#4a9eff"))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(2, 2, 60, 60)
    painter.setPen(QColor("#ffffff"))
    font = QFont("Arial", 32, QFont.Bold)
    painter.setFont(font)
    painter.drawText(px.rect(), Qt.AlignCenter, "T")
    painter.end()
    return QIcon(px)


class DropZone(QLabel):
    """Зона для перетаскивания или выбора файлов."""

    file_dropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(120)
        self.setCursor(Qt.PointingHandCursor)
        self._set_default_text()
        self.setObjectName("dropZone")

    def _set_default_text(self):
        self.setText("📁 Перетащите аудио/видео файл сюда\nили нажмите для выбора")

    def set_file(self, path):
        """Отображает имя выбранного файла."""
        self.setText(f"📄 {os.path.basename(path)}")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            ext_filter = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))
            path, _ = QFileDialog.getOpenFileName(
                self, "Выберите файл", "", f"Аудио/Видео ({ext_filter})"
            )
            if path:
                self.file_dropped.emit(path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        ext = os.path.splitext(path)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            self.file_dropped.emit(path)
        else:
            QMessageBox.warning(self, "Ошибка", f"Формат {ext} не поддерживается.")


class MainWindow(QMainWindow):
    """Главное окно приложения."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(800, 600)
        self.resize(920, 660)
        self.current_file = None
        self.worker = None
        self._recording = False
        self._rec_worker = None
        self._rec_start_time = 0.0

        # Персистентные настройки
        self._settings = QSettings("Transcriber", "Transcriber")
        self._is_dark = self._settings.value("theme", "light") == "dark"
        self._font_size = int(self._settings.value("font_size", _FONT_SIZE_DEFAULT))
        self._close_to_tray = True
        self._tray_hint_shown = self._settings.value("tray_hint_shown", False, type=bool)

        self._build_ui()
        self._build_menu_bar()
        self._build_status_bar()
        self._setup_tray()
        self._apply_theme(self._is_dark)
        self._apply_font_size(self._font_size)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        """Собирает все виджеты главного окна."""
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        # Верхняя часть: зона импорта ИЛИ превью плеера + настройки
        top = QHBoxLayout()
        self._file_stack = QStackedWidget()
        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self._on_file_selected)
        self._file_stack.addWidget(self.drop_zone)
        self.media_preview = MediaPreviewWidget()
        self.media_preview.file_selected.connect(self._on_file_selected)
        self._file_stack.addWidget(self.media_preview)
        top.addWidget(self._file_stack, stretch=2)

        settings = self._build_settings_panel()
        top.addLayout(settings, stretch=1)
        root.addLayout(top)

        # Индикатор прогресса (бесконечный)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(4)
        root.addWidget(self.progress_bar)

        # Поле результата
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("Результат транскрипции появится здесь...")
        self.result_text.installEventFilter(self)
        root.addWidget(self.result_text, stretch=1)

        # Нижние кнопки
        bottom = QHBoxLayout()
        self.copy_btn = QPushButton("📋 Копировать")
        self.copy_btn.clicked.connect(self._copy_result)
        self.copy_btn.setEnabled(False)
        self.save_btn = QPushButton("💾 Сохранить")
        self.save_btn.clicked.connect(self._save_result)
        self.save_btn.setEnabled(False)
        bottom.addStretch()
        bottom.addWidget(self.copy_btn)
        bottom.addWidget(self.save_btn)
        root.addLayout(bottom)

    def _build_settings_panel(self):
        """Создаёт панель настроек справа."""
        layout = QVBoxLayout()
        layout.setSpacing(8)

        layout.addWidget(QLabel("Модель:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_SIZES)
        self.model_combo.setCurrentText("large-v3")
        layout.addWidget(self.model_combo)

        layout.addWidget(QLabel("Язык:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(LANGUAGES.keys())
        layout.addWidget(self.lang_combo)

        fmt_box = QGroupBox("Формат вывода")
        fmt_layout = QVBoxLayout()
        self.fmt_plain = QRadioButton("Сплошной текст")
        self.fmt_timestamps = QRadioButton("С таймкодами")
        self.fmt_plain.setChecked(True)
        self.fmt_group = QButtonGroup()
        self.fmt_group.addButton(self.fmt_plain)
        self.fmt_group.addButton(self.fmt_timestamps)
        fmt_layout.addWidget(self.fmt_plain)
        fmt_layout.addWidget(self.fmt_timestamps)
        fmt_box.setLayout(fmt_layout)
        layout.addWidget(fmt_box)

        # Выбор микрофона
        layout.addWidget(QLabel("Микрофон:"))
        self.mic_combo = QComboBox()
        self.mic_combo.addItem("По умолчанию", userData=None)
        for dev in get_input_devices():
            self.mic_combo.addItem(dev["name"], userData=dev["index"])
        if not mic_available():
            self.mic_combo.setEnabled(False)
            self.mic_combo.setToolTip("Библиотека sounddevice не установлена")
        layout.addWidget(self.mic_combo)

        self.transcribe_btn = QPushButton("▶ Транскрибировать")
        self.transcribe_btn.setObjectName("primaryBtn")
        self.transcribe_btn.setEnabled(False)
        self.transcribe_btn.setMinimumHeight(40)
        self.transcribe_btn.clicked.connect(self._start_transcription)
        layout.addWidget(self.transcribe_btn)

        # Кнопка записи с микрофона
        self.record_btn = QPushButton("🎙 Запись с микрофона")
        self.record_btn.setObjectName("recordBtn")
        self.record_btn.setMinimumHeight(40)
        self.record_btn.clicked.connect(self._toggle_recording)
        if not mic_available():
            self.record_btn.setEnabled(False)
            self.record_btn.setToolTip("Библиотека sounddevice не установлена")
        layout.addWidget(self.record_btn)

        # Индикатор записи (скрыт по умолчанию)
        self._rec_indicator = QWidget()
        rec_layout = QHBoxLayout(self._rec_indicator)
        rec_layout.setContentsMargins(0, 4, 0, 4)
        rec_layout.setSpacing(8)
        self._rec_dot = QLabel("🔴")
        self._rec_dot.setStyleSheet("font-size: 16px;")
        rec_layout.addWidget(self._rec_dot)
        self._rec_timer_label = QLabel("0:00")
        self._rec_timer_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #d32f2f;"
        )
        rec_layout.addWidget(self._rec_timer_label)
        rec_layout.addStretch()
        self._rec_indicator.setVisible(False)
        layout.addWidget(self._rec_indicator)

        # Таймеры для анимации индикатора записи
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(500)
        self._pulse_timer.timeout.connect(self._pulse_indicator)

        self._rec_clock_timer = QTimer(self)
        self._rec_clock_timer.setInterval(1000)
        self._rec_clock_timer.timeout.connect(self._update_recording_timer)

        layout.addStretch()
        return layout

    def _build_menu_bar(self):
        """Создаёт строку меню с пунктами «Вид» и «Справка»."""
        menu_bar = self.menuBar()

        # --- Меню «Вид» ---
        view_menu = menu_bar.addMenu("Вид")

        self._dark_theme_action = QAction("Тёмная тема", self)
        self._dark_theme_action.setCheckable(True)
        self._dark_theme_action.setChecked(self._is_dark)
        self._dark_theme_action.triggered.connect(self._on_toggle_theme)
        view_menu.addAction(self._dark_theme_action)

        view_menu.addSeparator()

        zoom_in_action = QAction("Увеличить шрифт", self)
        zoom_in_action.setShortcut("Ctrl+=")
        zoom_in_action.triggered.connect(self._zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Уменьшить шрифт", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self._zoom_out)
        view_menu.addAction(zoom_out_action)

        zoom_reset_action = QAction("Сбросить масштаб", self)
        zoom_reset_action.setShortcut("Ctrl+0")
        zoom_reset_action.triggered.connect(self._zoom_reset)
        view_menu.addAction(zoom_reset_action)

        # --- Меню «Справка» ---
        help_menu = menu_bar.addMenu("Справка")
        env_check_action = help_menu.addAction("Проверить окружение\u2026")
        env_check_action.triggered.connect(self._show_env_check)

    def _show_env_check(self):
        """Открывает диалог проверки системного окружения."""
        from env_check import EnvCheckDialog
        dlg = EnvCheckDialog(self)
        dlg.exec()

    def _build_status_bar(self):
        """Настраивает статусную строку с информацией об устройстве."""
        device, compute = get_device_info()
        backend, _ = get_backend_info()
        badge = "🟢 GPU (CUDA)" if device == "cuda" else "🟡 CPU"
        self.status_label = QLabel(f"{badge} ({compute}) | {backend} | Готово")
        self.statusBar().addPermanentWidget(self.status_label)

    # ------------------------------------------------------------------ Системный трей

    def _setup_tray(self):
        """Создаёт иконку и контекстное меню системного трея."""
        # Используем иконку приложения, если есть, иначе генерируем
        app_icon = QApplication.instance().windowIcon()
        if app_icon.isNull():
            app_icon = _make_tray_icon()

        self._tray_icon = QSystemTrayIcon(app_icon, self)

        tray_menu = QMenu()
        show_action = tray_menu.addAction("Показать/Скрыть")
        show_action.triggered.connect(self._toggle_visibility)
        tray_menu.addSeparator()
        quit_action = tray_menu.addAction("Выход")
        quit_action.triggered.connect(self._quit_app)

        self._tray_icon.setContextMenu(tray_menu)
        self._tray_icon.activated.connect(self._on_tray_activated)
        self._tray_icon.setToolTip(APP_NAME)
        self._tray_icon.show()

    def _toggle_visibility(self):
        """Переключает видимость главного окна."""
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.activateWindow()
            self.raise_()

    def _on_tray_activated(self, reason):
        """Обработка двойного клика по иконке трея."""
        if reason == QSystemTrayIcon.DoubleClick:
            self._toggle_visibility()

    def _quit_app(self):
        """Полный выход из приложения (не сворачивание)."""
        self._close_to_tray = False
        if self._recording and self._rec_worker is not None:
            self._rec_worker.stop()
            self._rec_worker.wait(3000)
        self.media_preview.shutdown()
        self._tray_icon.hide()
        QApplication.instance().quit()

    # ------------------------------------------------------------------ Тема

    def _on_toggle_theme(self, checked: bool):
        """Переключает тёмную/светлую тему."""
        self._is_dark = checked
        self._apply_theme(checked)
        self._settings.setValue("theme", "dark" if checked else "light")

    def _apply_theme(self, dark: bool = False):
        """Применяет палитру и стили в зависимости от темы."""
        app = QApplication.instance()
        app.setStyle("Fusion")

        if dark:
            self._apply_dark_palette(app)
            self._apply_dark_stylesheet()
        else:
            self._apply_light_palette(app)
            self._apply_light_stylesheet()

    @staticmethod
    def _apply_light_palette(app: QApplication):
        """Светлая палитра (оригинальная)."""
        p = QPalette()
        dark = QColor(26, 26, 26)
        white = QColor(255, 255, 255)
        gray_bg = QColor(248, 249, 250)
        p.setColor(QPalette.Window, white)
        p.setColor(QPalette.WindowText, dark)
        p.setColor(QPalette.Base, white)
        p.setColor(QPalette.AlternateBase, gray_bg)
        p.setColor(QPalette.Text, dark)
        p.setColor(QPalette.Button, gray_bg)
        p.setColor(QPalette.ButtonText, dark)
        p.setColor(QPalette.PlaceholderText, QColor(120, 120, 120))
        p.setColor(QPalette.Highlight, QColor(74, 158, 255))
        p.setColor(QPalette.HighlightedText, white)
        app.setPalette(p)

    @staticmethod
    def _apply_dark_palette(app: QApplication):
        """Тёмная палитра."""
        p = QPalette()
        bg = QColor("#1e1e1e")
        surface = QColor("#2d2d2d")
        text = QColor("#e0e0e0")
        accent = QColor("#4a9eff")
        p.setColor(QPalette.Window, bg)
        p.setColor(QPalette.WindowText, text)
        p.setColor(QPalette.Base, surface)
        p.setColor(QPalette.AlternateBase, QColor("#353535"))
        p.setColor(QPalette.Text, text)
        p.setColor(QPalette.Button, surface)
        p.setColor(QPalette.ButtonText, text)
        p.setColor(QPalette.PlaceholderText, QColor(140, 140, 140))
        p.setColor(QPalette.Highlight, accent)
        p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        app.setPalette(p)

    def _apply_light_stylesheet(self):
        """Светлый стиль (оригинальный)."""
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #ffffff; color: #1a1a1a; }
            QLabel { font-size: 13px; color: #1a1a1a; }
            #dropZone {
                border: 2px dashed #b0b0b0;
                border-radius: 12px;
                background: #f8f9fa;
                color: #333333;
                font-size: 14px;
                padding: 20px;
            }
            #dropZone:hover { border-color: #4a9eff; background: #f0f6ff; color: #1a1a1a; }
            QPushButton {
                padding: 8px 16px; border-radius: 6px;
                border: 1px solid #ddd; background: #f8f9fa; font-size: 13px;
                color: #1a1a1a;
            }
            QPushButton:hover { background: #e9ecef; }
            QPushButton:disabled { color: #888888; background: #f0f0f0; }
            #primaryBtn {
                background: #4a9eff; color: #ffffff;
                border: none; font-weight: bold; font-size: 14px;
            }
            #primaryBtn:hover { background: #3a8eef; color: #ffffff; }
            #primaryBtn:disabled { background: #b0d4ff; color: #ffffff; }
            #recordBtn {
                background: #f8f9fa; color: #d32f2f;
                border: 1px solid #d32f2f; font-weight: bold; font-size: 14px;
            }
            #recordBtn:hover { background: #fce4ec; }
            #recordBtn:disabled { color: #888888; background: #f0f0f0; border-color: #ccc; }
            #recordingBtn {
                background: #d32f2f; color: #ffffff;
                border: none; font-weight: bold; font-size: 14px;
            }
            #recordingBtn:hover { background: #b71c1c; }
            QTextEdit {
                border: 1px solid #ddd; border-radius: 8px;
                padding: 12px; font-size: 13px;
                background: #fafafa; color: #1a1a1a;
            }
            QComboBox {
                padding: 6px 12px; border: 1px solid #ddd;
                border-radius: 6px; background: #ffffff; color: #1a1a1a;
                min-height: 1.2em;
            }
            QComboBox:hover { border-color: #4a9eff; }
            QComboBox QAbstractItemView {
                background: #ffffff;
                color: #1a1a1a;
                selection-background-color: #cfe8ff;
                selection-color: #1a1a1a;
                outline: 0;
            }
            QRadioButton { color: #1a1a1a; spacing: 8px; }
            QRadioButton::indicator { width: 16px; height: 16px; }
            QGroupBox {
                font-weight: bold; color: #1a1a1a;
                border: 1px solid #eee;
                border-radius: 8px; margin-top: 8px; padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 6px;
                color: #1a1a1a;
            }
            QMenuBar { background: #f8f9fa; color: #1a1a1a; border-bottom: 1px solid #eee; }
            QMenuBar::item:selected { background: #e9ecef; }
            QMenu { background: #ffffff; color: #1a1a1a; border: 1px solid #ddd; }
            QMenu::item:selected { background: #cfe8ff; color: #1a1a1a; }
            QStatusBar { border-top: 1px solid #eee; font-size: 12px; color: #1a1a1a; }
            QStatusBar QLabel { color: #1a1a1a; }
            QVideoWidget { background: #000000; border-radius: 8px; }
            QSlider::groove:horizontal { height: 6px; background: #e0e0e0; border-radius: 3px; }
            QSlider::handle:horizontal {
                width: 14px; margin: -5px 0; border-radius: 7px; background: #4a9eff;
            }
        """)

    def _apply_dark_stylesheet(self):
        """Тёмный стиль."""
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #1e1e1e; color: #e0e0e0; }
            QLabel { font-size: 13px; color: #e0e0e0; }
            #dropZone {
                border: 2px dashed #555555;
                border-radius: 12px;
                background: #2d2d2d;
                color: #cccccc;
                font-size: 14px;
                padding: 20px;
            }
            #dropZone:hover { border-color: #4a9eff; background: #353535; color: #e0e0e0; }
            QPushButton {
                padding: 8px 16px; border-radius: 6px;
                border: 1px solid #404040; background: #2d2d2d; font-size: 13px;
                color: #e0e0e0;
            }
            QPushButton:hover { background: #353535; }
            QPushButton:disabled { color: #666666; background: #252525; }
            #primaryBtn {
                background: #4a9eff; color: #ffffff;
                border: none; font-weight: bold; font-size: 14px;
            }
            #primaryBtn:hover { background: #3a8eef; color: #ffffff; }
            #primaryBtn:disabled { background: #2a5a8f; color: #888888; }
            #recordBtn {
                background: #2d2d2d; color: #ef5350;
                border: 1px solid #ef5350; font-weight: bold; font-size: 14px;
            }
            #recordBtn:hover { background: #3a2020; }
            #recordBtn:disabled { color: #666666; background: #252525; border-color: #404040; }
            #recordingBtn {
                background: #d32f2f; color: #ffffff;
                border: none; font-weight: bold; font-size: 14px;
            }
            #recordingBtn:hover { background: #b71c1c; }
            QTextEdit {
                border: 1px solid #404040; border-radius: 8px;
                padding: 12px; font-size: 13px;
                background: #2d2d2d; color: #e0e0e0;
            }
            QComboBox {
                padding: 6px 12px; border: 1px solid #404040;
                border-radius: 6px; background: #2d2d2d; color: #e0e0e0;
                min-height: 1.2em;
            }
            QComboBox:hover { border-color: #4a9eff; }
            QComboBox QAbstractItemView {
                background: #2d2d2d;
                color: #e0e0e0;
                selection-background-color: #3a5a7f;
                selection-color: #ffffff;
                outline: 0;
            }
            QRadioButton { color: #e0e0e0; spacing: 8px; }
            QRadioButton::indicator { width: 16px; height: 16px; }
            QGroupBox {
                font-weight: bold; color: #e0e0e0;
                border: 1px solid #404040;
                border-radius: 8px; margin-top: 8px; padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 6px;
                color: #e0e0e0;
            }
            QMenuBar { background: #2d2d2d; color: #e0e0e0; border-bottom: 1px solid #404040; }
            QMenuBar::item:selected { background: #353535; }
            QMenu { background: #2d2d2d; color: #e0e0e0; border: 1px solid #404040; }
            QMenu::item:selected { background: #3a5a7f; color: #ffffff; }
            QStatusBar { border-top: 1px solid #404040; font-size: 12px; color: #e0e0e0; }
            QStatusBar QLabel { color: #e0e0e0; }
            QVideoWidget { background: #000000; border-radius: 8px; }
            QSlider::groove:horizontal { height: 6px; background: #404040; border-radius: 3px; }
            QSlider::handle:horizontal {
                width: 14px; margin: -5px 0; border-radius: 7px; background: #4a9eff;
            }
        """)

    # ------------------------------------------------------------------ Масштаб шрифта

    def _apply_font_size(self, size: int):
        """Устанавливает размер шрифта для поля результата."""
        size = max(_FONT_SIZE_MIN, min(_FONT_SIZE_MAX, size))
        self._font_size = size
        font = self.result_text.font()
        font.setPointSize(size)
        self.result_text.setFont(font)

    def _zoom_in(self):
        """Увеличить шрифт на 1pt."""
        if self._font_size < _FONT_SIZE_MAX:
            self._apply_font_size(self._font_size + 1)
            self._save_font_size()
            self.status_label.setText(f"Размер шрифта: {self._font_size}pt")

    def _zoom_out(self):
        """Уменьшить шрифт на 1pt."""
        if self._font_size > _FONT_SIZE_MIN:
            self._apply_font_size(self._font_size - 1)
            self._save_font_size()
            self.status_label.setText(f"Размер шрифта: {self._font_size}pt")

    def _zoom_reset(self):
        """Сбросить масштаб шрифта к значению по умолчанию."""
        self._apply_font_size(_FONT_SIZE_DEFAULT)
        self._save_font_size()
        self.status_label.setText(f"Размер шрифта сброшен: {self._font_size}pt")

    def _save_font_size(self):
        """Сохраняет размер шрифта в настройках."""
        self._settings.setValue("font_size", self._font_size)

    def eventFilter(self, obj, event):
        """Перехватывает Ctrl+Wheel на поле результата для масштабирования."""
        if obj is self.result_text and event.type() == event.Type.Wheel:
            if event.modifiers() & Qt.ControlModifier:
                delta = event.angleDelta().y()
                if delta > 0:
                    self._zoom_in()
                elif delta < 0:
                    self._zoom_out()
                return True
        return super().eventFilter(obj, event)

    # --- Обработчики событий ---

    def _is_transcribing(self):
        """Проверяет, идёт ли транскрипция в данный момент."""
        return self.worker is not None and self.worker.isRunning()

    def closeEvent(self, event: QCloseEvent):
        """При закрытии сворачиваем в трей, если _close_to_tray=True."""
        if self._close_to_tray and QSystemTrayIcon.isSystemTrayAvailable():
            event.ignore()
            self.hide()
            if not self._tray_hint_shown:
                self._tray_icon.showMessage(
                    APP_NAME,
                    "Приложение свёрнуто в трей",
                    QSystemTrayIcon.Information,
                    2000,
                )
                self._tray_hint_shown = True
                self._settings.setValue("tray_hint_shown", True)
        else:
            # Полное закрытие
            if self._recording and self._rec_worker is not None:
                self._rec_worker.stop()
                self._rec_worker.wait(3000)
            self.media_preview.shutdown()
            self._tray_icon.hide()
            super().closeEvent(event)

    def _on_file_selected(self, path):
        """Вызывается при выборе файла (drag & drop или диалог)."""
        if self._is_transcribing():
            return
        self.current_file = path
        # Пока файла нет — страница 0 (drag & drop); после выбора — плеер + «Открыть» снизу
        self._file_stack.setCurrentIndex(1)
        self.media_preview.load(path)
        self.transcribe_btn.setEnabled(True)
        self.status_label.setText(f"Файл: {os.path.basename(path)}")

    def _start_transcription(self, file_path=None, skip_convert=False):
        """Запускает транскрипцию в фоновом потоке.

        Args:
            file_path: путь к файлу (None = self.current_file)
            skip_convert: пропустить конвертацию ffmpeg (файл уже WAV 16 kHz моно)
        """
        path = file_path or self.current_file
        if not path:
            return
        self.transcribe_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        self.drop_zone.setAcceptDrops(False)
        self.media_preview.set_interactive(False)
        self.progress_bar.setVisible(True)
        self.result_text.clear()
        self.copy_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

        lang_code = LANGUAGES.get(self.lang_combo.currentText(), "auto")
        fmt = "timestamps" if self.fmt_timestamps.isChecked() else "plain"

        self.worker = TranscribeWorker(
            path, self.model_combo.currentText(), lang_code, fmt,
            skip_convert=skip_convert,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, message):
        self.status_label.setText(message)

    def _on_finished(self, text, lang_info):
        """Обработка успешного завершения транскрипции."""
        self.progress_bar.setVisible(False)
        self.transcribe_btn.setEnabled(bool(self.current_file))
        self.record_btn.setEnabled(mic_available())
        self.drop_zone.setAcceptDrops(True)
        self.media_preview.set_interactive(True)
        self.result_text.setPlainText(text)
        self.copy_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

        detected = lang_info.get("language", "?")
        prob = lang_info.get("probability", 0)
        device, compute = get_device_info()
        badge = "🟢 GPU" if device == "cuda" else "🟡 CPU"
        self.status_label.setText(
            f"{badge} ({compute}) | Язык: {detected} ({prob:.0%}) | Готово"
        )

    def _on_error(self, error_msg):
        """Обработка ошибки транскрипции."""
        self.progress_bar.setVisible(False)
        self.transcribe_btn.setEnabled(bool(self.current_file))
        self.record_btn.setEnabled(mic_available())
        self.drop_zone.setAcceptDrops(True)
        self.media_preview.set_interactive(True)
        self.status_label.setText("❌ Ошибка")
        QMessageBox.critical(self, "Ошибка транскрипции", error_msg)

    # --- Запись с микрофона ---

    def _toggle_recording(self):
        """Начинает или останавливает запись с микрофона."""
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        """Запускает запись с микрофона."""
        device_index = self.mic_combo.currentData()

        self._recording = True
        self._rec_start_time = time.monotonic()

        # Обновляем UI
        self.record_btn.setText("⏹ Остановить запись")
        self.record_btn.setObjectName("recordingBtn")
        # Принудительно обновляем стиль после смены objectName
        self.record_btn.setStyleSheet(self.record_btn.styleSheet())
        self._apply_theme(self._is_dark)

        self.transcribe_btn.setEnabled(False)
        self.drop_zone.setAcceptDrops(False)
        self.media_preview.set_interactive(False)

        # Показываем индикатор записи
        self._rec_timer_label.setText("0:00")
        self._rec_dot.setVisible(True)
        self._rec_indicator.setVisible(True)
        self._pulse_timer.start()
        self._rec_clock_timer.start()

        # Запускаем запись
        self._rec_worker = MicRecorderWorker(device_index=device_index)
        self._rec_worker.finished.connect(self._on_recording_finished)
        self._rec_worker.error.connect(self._on_recording_error)
        self._rec_worker.start()

        self.status_label.setText("🎙 Запись с микрофона...")

    def _stop_recording(self):
        """Останавливает запись с микрофона."""
        if self._rec_worker is not None:
            self._rec_worker.stop()
        self._pulse_timer.stop()
        self._rec_clock_timer.stop()
        self.status_label.setText("Обработка записи...")

    def _on_recording_finished(self, wav_path):
        """Обработка завершения записи: запускает транскрипцию WAV."""
        self._recording = False
        self._rec_worker = None

        # Скрываем индикатор и восстанавливаем кнопку
        self._rec_indicator.setVisible(False)
        self._pulse_timer.stop()
        self._rec_clock_timer.stop()
        self.record_btn.setText("🎙 Запись с микрофона")
        self.record_btn.setObjectName("recordBtn")
        self.record_btn.setStyleSheet(self.record_btn.styleSheet())
        self._apply_theme(self._is_dark)

        self.drop_zone.setAcceptDrops(True)
        self.media_preview.set_interactive(True)

        # Запускаем транскрипцию записанного файла (без конвертации ffmpeg)
        self._start_transcription(file_path=wav_path, skip_convert=True)

    def _on_recording_error(self, error_msg):
        """Обработка ошибки записи с микрофона."""
        self._recording = False
        self._rec_worker = None

        # Скрываем индикатор и восстанавливаем UI
        self._rec_indicator.setVisible(False)
        self._pulse_timer.stop()
        self._rec_clock_timer.stop()
        self.record_btn.setText("🎙 Запись с микрофона")
        self.record_btn.setObjectName("recordBtn")
        self.record_btn.setStyleSheet(self.record_btn.styleSheet())
        self._apply_theme(self._is_dark)

        self.transcribe_btn.setEnabled(bool(self.current_file))
        self.drop_zone.setAcceptDrops(True)
        self.media_preview.set_interactive(True)

        self.status_label.setText("❌ Ошибка записи")
        QMessageBox.critical(self, "Ошибка записи", error_msg)

    def _update_recording_timer(self):
        """Обновляет таймер длительности записи каждую секунду."""
        elapsed = int(time.monotonic() - self._rec_start_time)
        minutes = elapsed // 60
        seconds = elapsed % 60
        self._rec_timer_label.setText(f"{minutes}:{seconds:02d}")

    def _pulse_indicator(self):
        """Переключает видимость красной точки для эффекта пульсации."""
        self._rec_dot.setVisible(not self._rec_dot.isVisible())

    def _copy_result(self):
        QApplication.clipboard().setText(self.result_text.toPlainText())
        self.status_label.setText("✔ Текст скопирован в буфер обмена")

    def _save_result(self):
        default_name = ""
        if self.current_file:
            base = os.path.splitext(os.path.basename(self.current_file))[0]
            default_name = f"{base}_transcription.txt"
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить транскрипцию", default_name,
            "Текстовый файл (*.txt);;Все файлы (*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.result_text.toPlainText())
            self.status_label.setText(f"✔ Сохранено: {os.path.basename(path)}")
