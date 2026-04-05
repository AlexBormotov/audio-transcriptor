"""
Скрипт сборки Windows .exe через PyInstaller.

Запуск на Windows:
    python build_exe.py

Результат: папка dist/Transcriber/ с готовым .exe и всеми зависимостями.
Затем installer.iss (Inno Setup) собирает установщик из этой папки.

Требования:
    pip install pyinstaller
"""

import os
import sys


def build():
    # PyInstaller нужно импортировать после проверки наличия
    try:
        import PyInstaller.__main__
    except ImportError:
        print("PyInstaller не установлен. Установите: pip install pyinstaller")
        sys.exit(1)
    # Важно: GUI зависит от PySide6, без него exe соберётся некорректно.
    # Проверяем заранее и падаем с понятной инструкцией.
    try:
        import PySide6  # noqa: F401
    except ImportError:
        print("PySide6 не установлен в текущем Python-окружении.")
        print("Установите зависимости в активном venv: pip install -r requirements.txt")
        sys.exit(1)

    app_name = "Transcriber"
    main_script = "main.py"
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Неиспользуемые Qt/PySide6-модули — исключаем для ускорения сборки и уменьшения размера.
    pyside6_excludes = [
        "PySide6.QtWebEngine", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebChannel", "PySide6.QtWebSockets",
        "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DInput",
        "PySide6.Qt3DLogic", "PySide6.Qt3DAnimation", "PySide6.Qt3DExtras",
        "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets",
        "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtSensors",
        "PySide6.QtSerialPort", "PySide6.QtPositioning", "PySide6.QtLocation",
        "PySide6.QtCharts", "PySide6.QtDataVisualization",
        "PySide6.QtQuick", "PySide6.QtQuickWidgets", "PySide6.QtQml",
        "PySide6.QtRemoteObjects", "PySide6.QtScxml", "PySide6.QtSql",
        "PySide6.QtTest", "PySide6.QtXml", "PySide6.QtDesigner",
        "PySide6.QtHelp", "PySide6.QtPdf", "PySide6.QtPdfWidgets",
        "PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets",
        "PySide6.QtSpatialAudio", "PySide6.QtHttpServer",
    ]

    args = [
        os.path.join(base_dir, main_script),
        f"--name={app_name}",
        "--windowed",
        "--onedir",
        "--noconfirm",
        # Скрытые импорты, которые PyInstaller может не найти
        "--hidden-import=transcriber",
        "--hidden-import=constants",
        "--hidden-import=worker",
        "--hidden-import=main_window",
        "--hidden-import=faster_whisper",
        "--hidden-import=torch",
        "--hidden-import=ctranslate2",
        # Собрать данные PySide6 (плагины, стили)
        "--collect-all=PySide6",
    ]

    # Исключаем тяжёлые неиспользуемые модули
    for mod in pyside6_excludes:
        args.append(f"--exclude-module={mod}")

    # Добавить иконку если существует
    icon_path = os.path.join(base_dir, "icon.ico")
    if os.path.exists(icon_path):
        args.append(f"--icon={icon_path}")
        print(f"[OK] Иконка: {icon_path}")
    else:
        print("[INFO] icon.ico не найден, сборка без иконки.")

    print(f"[INFO] Сборка {app_name}...")
    print(f"[INFO] Команда: pyinstaller {' '.join(args[1:])}")
    print()

    PyInstaller.__main__.run(args)

    dist_path = os.path.join(base_dir, "dist", app_name)
    if os.path.isdir(dist_path):
        print(f"\n[OK] Сборка завершена: {dist_path}")
        print(f"[OK] Запуск: {os.path.join(dist_path, app_name + '.exe')}")
        print("[INFO] Для создания установщика запустите Inno Setup с installer.iss")
    else:
        print("\n[ОШИБКА] Папка dist не создана. Проверьте ошибки выше.")


if __name__ == "__main__":
    build()
