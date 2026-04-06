# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = ['transcriber', 'constants', 'worker', 'main_window', 'media_preview', 'PySide6.QtMultimedia', 'PySide6.QtMultimediaWidgets', 'faster_whisper', 'torch', 'ctranslate2']
tmp_ret = collect_all('PySide6')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['D:\\Projects\\!Python\\CursorProjects\\audio-transcriptor\\main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6.QtWebEngine', 'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets', 'PySide6.QtWebChannel', 'PySide6.QtWebSockets', 'PySide6.Qt3DCore', 'PySide6.Qt3DRender', 'PySide6.Qt3DInput', 'PySide6.Qt3DLogic', 'PySide6.Qt3DAnimation', 'PySide6.Qt3DExtras', 'PySide6.QtBluetooth', 'PySide6.QtNfc', 'PySide6.QtSensors', 'PySide6.QtSerialPort', 'PySide6.QtPositioning', 'PySide6.QtLocation', 'PySide6.QtCharts', 'PySide6.QtDataVisualization', 'PySide6.QtQuick', 'PySide6.QtQuickWidgets', 'PySide6.QtQml', 'PySide6.QtRemoteObjects', 'PySide6.QtScxml', 'PySide6.QtSql', 'PySide6.QtTest', 'PySide6.QtXml', 'PySide6.QtDesigner', 'PySide6.QtHelp', 'PySide6.QtPdf', 'PySide6.QtPdfWidgets', 'PySide6.QtOpenGL', 'PySide6.QtOpenGLWidgets', 'PySide6.QtSpatialAudio', 'PySide6.QtHttpServer'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Transcriber',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Transcriber',
)
