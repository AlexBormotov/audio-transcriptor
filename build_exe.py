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

    args = [
        os.path.join(base_dir, main_script),
        f"--name={app_name}",
        # --windowed: без консольного окна (GUI-приложение)
        "--windowed",
        # --onedir: папка с .exe + зависимости (быстрый запуск)
        "--onedir",
        "--noconfirm",
        "--clean",
        # Скрытые импорты, которые PyInstaller может не найти
        "--hidden-import=transcriber",
        "--hidden-import=constants",
        "--hidden-import=worker",
        "--hidden-import=main_window",
        "--hidden-import=faster_whisper",
        "--hidden-import=torch",
        "--hidden-import=ctranslate2",
        # Собрать все данные PySide6 (плагины, стили)
        "--collect-all=PySide6",
    ]

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
