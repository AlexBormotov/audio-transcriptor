"""
Точка входа десктопного приложения «Транскрибатор».

Запуск: python main.py
Создаёт PySide6-окно с drag & drop загрузкой файлов,
настройками модели/языка и отображением результата.
"""

import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon, QPalette, QColor

from constants import APP_NAME
from main_window import MainWindow


def _apply_light_palette(app: QApplication) -> None:
    """
    Явная светлая палитра: на Windows 11 с тёмной темой система может отдавать
    светлый текст, а QSS задаёт светлый фон — получается «белое на белом».
    Fusion + палитра фиксируют цвет текста и списков QComboBox.
    """
    app.setStyle("Fusion")
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


def main():
    app = QApplication(sys.argv)
    _apply_light_palette(app)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("Transcriber")

    # Иконка приложения (если есть файл icon.png/icon.ico рядом)
    import os
    icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
