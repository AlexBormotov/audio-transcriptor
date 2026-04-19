"""
Точка входа десктопного приложения «Транскрибатор».

Запуск: python main.py
Создаёт PySide6-окно с drag & drop загрузкой файлов,
настройками модели/языка и отображением результата.
"""

import sys

from PySide6.QtWidgets import QApplication, QSplashScreen
from PySide6.QtGui import QIcon, QPixmap, QColor
from PySide6.QtCore import Qt

from constants import APP_NAME
from main_window import MainWindow


def _show_loading_splash() -> QSplashScreen:
    """
    Показывает простой стартовый экран:
    символ загрузки + надпись Loading...
    """
    pixmap = QPixmap(320, 120)
    pixmap.fill(QColor("#f8f9fa"))
    splash = QSplashScreen(pixmap)
    splash.showMessage("⏳ Loading...", Qt.AlignCenter, QColor("#1a1a1a"))
    splash.show()
    return splash


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("Transcriber")
    splash = _show_loading_splash()
    app.processEvents()

    # Иконка приложения (если есть файл icon.png/icon.ico рядом)
    import os
    icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = MainWindow()
    window.show()
    app.processEvents()
    splash.finish(window)
    splash.close()
    splash.deleteLater()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
