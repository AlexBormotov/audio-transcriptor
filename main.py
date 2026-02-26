"""
Точка входа десктопного приложения «Транскрибатор».

Запуск: python main.py
Создаёт PySide6-окно с drag & drop загрузкой файлов,
настройками модели/языка и отображением результата.
"""

import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from constants import APP_NAME
from main_window import MainWindow


def main():
    app = QApplication(sys.argv)
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
