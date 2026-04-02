"""
Главное окно десктопного приложения-транскрибатора.

Содержит зону drag & drop, настройки модели/языка/формата,
область результата и кнопки копирования/сохранения.
"""

import os

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QRadioButton, QTextEdit,
    QLabel, QFileDialog, QProgressBar,
    QButtonGroup, QGroupBox, QMessageBox, QApplication,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from constants import MODEL_SIZES, LANGUAGES, SUPPORTED_EXTENSIONS, APP_NAME
from worker import TranscribeWorker
from transcriber import get_device_info, get_backend_info


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
        self._build_ui()
        self._build_status_bar()
        self._apply_styles()

    def _build_ui(self):
        """Собирает все виджеты главного окна."""
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        # Верхняя часть: drop zone + настройки
        top = QHBoxLayout()
        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self._on_file_selected)
        top.addWidget(self.drop_zone, stretch=2)

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
        self.model_combo.setCurrentText("base")
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

        self.transcribe_btn = QPushButton("▶ Транскрибировать")
        self.transcribe_btn.setObjectName("primaryBtn")
        self.transcribe_btn.setEnabled(False)
        self.transcribe_btn.setMinimumHeight(40)
        self.transcribe_btn.clicked.connect(self._start_transcription)
        layout.addWidget(self.transcribe_btn)

        layout.addStretch()
        return layout

    def _build_status_bar(self):
        """Настраивает статусную строку с информацией об устройстве."""
        device, compute = get_device_info()
        backend, _ = get_backend_info()
        badge = "🟢 GPU (CUDA)" if device == "cuda" else "🟡 CPU"
        self.status_label = QLabel(f"{badge} ({compute}) | {backend} | Готово")
        self.statusBar().addPermanentWidget(self.status_label)

    def _apply_styles(self):
        """Применяет современный плоский стиль ко всему окну."""
        # Везде явный тёмный текст (#1a1a1a), иначе на Windows с тёмной темой
        # виджеты могут рисовать светлый текст поверх светлого фона из QSS.
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
            /* Выпадающий список — отдельное представление; без color текст бывает белым */
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
            QStatusBar { border-top: 1px solid #eee; font-size: 12px; color: #1a1a1a; }
            QStatusBar QLabel { color: #1a1a1a; }
        """)

    # --- Обработчики событий ---

    def _on_file_selected(self, path):
        """Вызывается при выборе файла (drag & drop или диалог)."""
        self.current_file = path
        self.drop_zone.set_file(path)
        self.transcribe_btn.setEnabled(True)
        self.status_label.setText(f"Файл: {os.path.basename(path)}")

    def _start_transcription(self):
        """Запускает транскрипцию в фоновом потоке."""
        if not self.current_file:
            return
        self.transcribe_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.result_text.clear()
        self.copy_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

        lang_code = LANGUAGES.get(self.lang_combo.currentText(), "auto")
        fmt = "timestamps" if self.fmt_timestamps.isChecked() else "plain"

        self.worker = TranscribeWorker(
            self.current_file, self.model_combo.currentText(), lang_code, fmt
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
        self.transcribe_btn.setEnabled(True)
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
        self.transcribe_btn.setEnabled(True)
        self.status_label.setText("❌ Ошибка")
        QMessageBox.critical(self, "Ошибка транскрипции", error_msg)

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
