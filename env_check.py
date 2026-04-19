"""
Диалог проверки внешних компонентов (окружения).

Проверяет наличие ffmpeg, NVIDIA GPU, PyTorch, sounddevice
и Visual C++ Redistributable. Показывает результаты в модальном окне
со ссылками на загрузку недостающих компонентов.
"""

import subprocess
import sys

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QVBoxLayout, QWidget,
)


def check_environment():
    """Проверяет системные зависимости и возвращает список результатов.

    Returns:
        list[dict]: список словарей с ключами:
            component, status, message, help_url, help_text
    """
    results = []
    results.append(_check_ffmpeg())
    results.append(_check_pytorch())
    results.append(_check_nvidia_gpu())
    results.append(_check_sounddevice())
    if sys.platform == "win32":
        results.append(_check_vc_redist())
    return results


def _check_ffmpeg():
    """Проверяет наличие ffmpeg в PATH."""
    try:
        kwargs = dict(
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(["ffmpeg", "-version"], **kwargs)
        if result.returncode == 0:
            # Первая строка: "ffmpeg version X.Y.Z ..."
            first_line = result.stdout.strip().split("\n")[0]
            return {
                "component": "ffmpeg",
                "status": "ok",
                "message": f"Найден: {first_line}",
                "help_url": "https://ffmpeg.org/download.html",
                "help_text": "Скачать ffmpeg",
            }
        return {
            "component": "ffmpeg",
            "status": "missing",
            "message": "ffmpeg вернул ошибку (код возврата != 0)",
            "help_url": "https://ffmpeg.org/download.html",
            "help_text": "Скачать ffmpeg",
        }
    except FileNotFoundError:
        return {
            "component": "ffmpeg",
            "status": "missing",
            "message": "ffmpeg не найден в PATH",
            "help_url": "https://ffmpeg.org/download.html",
            "help_text": "Скачать ffmpeg",
        }
    except Exception as exc:
        return {
            "component": "ffmpeg",
            "status": "missing",
            "message": f"Ошибка при проверке ffmpeg: {exc}",
            "help_url": "https://ffmpeg.org/download.html",
            "help_text": "Скачать ffmpeg",
        }


def _check_pytorch():
    """Проверяет наличие и версию PyTorch."""
    try:
        import torch
    except ImportError:
        return {
            "component": "PyTorch",
            "status": "missing",
            "message": "PyTorch не установлен",
            "help_url": "https://pytorch.org/get-started/locally/",
            "help_text": "Установить PyTorch",
        }

    version = torch.__version__
    cuda_version = getattr(torch.version, "cuda", None)
    if cuda_version:
        label = f"{version} (CUDA {cuda_version})"
    else:
        label = f"{version} (только CPU)"
    return {
        "component": "PyTorch",
        "status": "ok",
        "message": label,
        "help_url": "https://pytorch.org/get-started/locally/",
        "help_text": "Страница PyTorch",
    }


def _check_nvidia_gpu():
    """Проверяет доступность NVIDIA GPU через torch.cuda."""
    try:
        import torch
    except ImportError:
        return {
            "component": "NVIDIA GPU",
            "status": "missing",
            "message": "Невозможно проверить — PyTorch не установлен",
            "help_url": "https://pytorch.org/get-started/locally/",
            "help_text": "Установить PyTorch",
        }

    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "неизвестная модель"
        return {
            "component": "NVIDIA GPU",
            "status": "ok",
            "message": f"{name}",
            "help_url": "https://developer.nvidia.com/cuda-downloads",
            "help_text": "CUDA Toolkit",
        }

    # torch импортируется, но CUDA недоступна
    cuda_version = getattr(torch.version, "cuda", None)
    if cuda_version is None:
        msg = "PyTorch собран без поддержки CUDA (CPU-only сборка)"
    else:
        msg = "CUDA недоступна (нет совместимого GPU или драйверов)"
    return {
        "component": "NVIDIA GPU",
        "status": "warning",
        "message": f"{msg} — будет использоваться CPU",
        "help_url": "https://developer.nvidia.com/cuda-downloads",
        "help_text": "CUDA Toolkit",
    }


def _check_sounddevice():
    """Проверяет наличие sounddevice (доступ к микрофону)."""
    try:
        import sounddevice as sd
    except ImportError:
        return {
            "component": "Микрофон (sounddevice)",
            "status": "missing",
            "message": "sounddevice не установлен — pip install sounddevice",
            "help_url": "https://python-sounddevice.readthedocs.io/",
            "help_text": "Документация sounddevice",
        }

    try:
        devices = sd.query_devices()
        input_devices = [d for d in devices if d.get("max_input_channels", 0) > 0]
        if input_devices:
            return {
                "component": "Микрофон (sounddevice)",
                "status": "ok",
                "message": f"Найдено устройств ввода: {len(input_devices)}",
                "help_url": "https://python-sounddevice.readthedocs.io/",
                "help_text": "Документация sounddevice",
            }
        return {
            "component": "Микрофон (sounddevice)",
            "status": "warning",
            "message": "Устройства ввода не обнаружены",
            "help_url": "https://python-sounddevice.readthedocs.io/",
            "help_text": "Документация sounddevice",
        }
    except Exception as exc:
        return {
            "component": "Микрофон (sounddevice)",
            "status": "warning",
            "message": f"Ошибка при опросе устройств: {exc}",
            "help_url": "https://python-sounddevice.readthedocs.io/",
            "help_text": "Документация sounddevice",
        }


def _check_vc_redist():
    """Проверяет Microsoft Visual C++ Redistributable (только Windows).

    Если PyTorch загружается нормально, считаем что VC++ Redist есть,
    т.к. torch зависит от этих библиотек.
    """
    try:
        import torch  # noqa: F401
        return {
            "component": "Visual C++ Redistributable",
            "status": "ok",
            "message": "Установлен (PyTorch загружен успешно)",
            "help_url": "https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist",
            "help_text": "Скачать VC++ Redist",
        }
    except ImportError:
        return {
            "component": "Visual C++ Redistributable",
            "status": "warning",
            "message": "Невозможно проверить — PyTorch не установлен",
            "help_url": "https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist",
            "help_text": "Скачать VC++ Redist",
        }
    except OSError:
        return {
            "component": "Visual C++ Redistributable",
            "status": "missing",
            "message": "Отсутствует или повреждён — требуется для PyTorch",
            "help_url": "https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist",
            "help_text": "Скачать VC++ Redist",
        }


_STATUS_ICONS = {
    "ok": "\u2705",       # ✅
    "warning": "\u26a0\ufe0f",  # ⚠️
    "missing": "\u274c",   # ❌
}


class EnvCheckDialog(QDialog):
    """Модальный диалог проверки системного окружения."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Проверка окружения")
        self.setMinimumSize(500, 300)
        self.resize(560, 400)
        self._build_ui()
        self._apply_styles()

    def _build_ui(self):
        """Собирает интерфейс диалога."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Заголовок
        title = QLabel("Проверка системных компонентов")
        title.setObjectName("envTitle")
        layout.addWidget(title)

        # Таблица результатов
        results_widget = QWidget()
        grid = QGridLayout(results_widget)
        grid.setSpacing(8)
        grid.setContentsMargins(0, 8, 0, 8)

        checks = check_environment()

        for row, item in enumerate(checks):
            icon = _STATUS_ICONS.get(item["status"], "")
            icon_label = QLabel(icon)
            icon_label.setFixedWidth(28)
            icon_label.setAlignment(Qt.AlignCenter)
            grid.addWidget(icon_label, row, 0)

            text = f"<b>{item['component']}</b> — {item['message']}"
            text_label = QLabel(text)
            text_label.setWordWrap(True)
            text_label.setTextFormat(Qt.RichText)
            grid.addWidget(text_label, row, 1)

            if item.get("help_url"):
                link_btn = QPushButton(item.get("help_text", "Ссылка"))
                link_btn.setCursor(Qt.PointingHandCursor)
                link_btn.setObjectName("linkBtn")
                url = item["help_url"]
                link_btn.clicked.connect(
                    lambda checked=False, u=url: QDesktopServices.openUrl(QUrl(u))
                )
                grid.addWidget(link_btn, row, 2)

        # Растяжка колонки с текстом
        grid.setColumnStretch(1, 1)

        layout.addWidget(results_widget)
        layout.addStretch()

        # Кнопка «Закрыть»
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Закрыть")
        close_btn.setMinimumWidth(100)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _apply_styles(self):
        """Стилизация диалога под светлую тему приложения."""
        self.setStyleSheet("""
            QDialog { background: #ffffff; color: #1a1a1a; }
            QLabel { font-size: 13px; color: #1a1a1a; }
            #envTitle { font-size: 15px; font-weight: bold; color: #1a1a1a; }
            QPushButton {
                padding: 6px 14px; border-radius: 6px;
                border: 1px solid #ddd; background: #f8f9fa;
                font-size: 12px; color: #1a1a1a;
            }
            QPushButton:hover { background: #e9ecef; }
            #linkBtn {
                border: none; background: transparent;
                color: #4a9eff; font-size: 12px; text-decoration: underline;
            }
            #linkBtn:hover { color: #3a8eef; }
        """)
