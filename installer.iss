; Inno Setup Script — установщик «Транскрибатор» для Windows x64
;
; Как использовать:
; 1. Сначала соберите .exe: python build_exe.py
; 2. Скачайте Inno Setup: https://jrsoftware.org/isdl.php
; 3. Откройте этот файл в Inno Setup Compiler
; 4. Нажмите Build → Compile
; 5. Готовый установщик появится в папке installer_output/

[Setup]
; Название и версия приложения
AppName=Транскрибатор
AppVersion=1.0.0
AppVerName=Транскрибатор 1.0.0
AppPublisher=Transcriber
AppPublisherURL=https://github.com/niceguy135/SpeechRecognition

; Папка установки по умолчанию
DefaultDirName={autopf}\Transcriber
DefaultGroupName=Транскрибатор

; Файл установщика
OutputDir=installer_output
OutputBaseFilename=TranscriberSetup_x64

; Только 64-битные системы
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; Сжатие LZMA2 для минимального размера
Compression=lzma2
SolidCompression=yes

; Иконка установщика (если есть icon.ico)
; SetupIconFile=icon.ico
; UninstallDisplayIcon={app}\Transcriber.exe

; Лицензия (если есть)
; LicenseFile=LICENSE

; Минимальная версия Windows
MinVersion=10.0

; Страница выбора директории и подтверждения
DisableDirPage=no
DisableProgramGroupPage=yes

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
; Галочка «Создать ярлык на рабочем столе»
Name: "desktopicon"; Description: "Создать ярлык на рабочем столе"; GroupDescription: "Дополнительно:"; Flags: checked

[Files]
; Копируем всё из папки dist/Transcriber/ в папку установки
Source: "dist\Transcriber\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Ярлык в меню Пуск
Name: "{group}\Транскрибатор"; Filename: "{app}\Transcriber.exe"
; Ярлык на рабочем столе (если выбран)
Name: "{autodesktop}\Транскрибатор"; Filename: "{app}\Transcriber.exe"; Tasks: desktopicon
; Ярлык удаления в меню Пуск
Name: "{group}\Удалить Транскрибатор"; Filename: "{uninstallexe}"

[Run]
; Предложение запустить после установки
Filename: "{app}\Transcriber.exe"; Description: "Запустить Транскрибатор"; Flags: nowait postinstall skipifsilent
