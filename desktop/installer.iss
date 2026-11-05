; Micrograph Maker の Windows 用インストーラー（Inno Setup 6）。GitHub Actions がリポジトリのいちばん上で
;   iscc /DAppVersion=0.11.0 desktop\installer.iss
; を実行して installer\MicrographMaker-0.11.0-windows-setup.exe を作る。管理者の権限は要らない（自分のユーザーのフォルダに入れる）。
#ifndef AppVersion
  #define AppVersion "0"
#endif

[Setup]
AppId={{7DF8176F-CA80-416B-B33F-14EBE6C506B0}
AppName=Micrograph Maker
AppVersion={#AppVersion}
AppVerName=Micrograph Maker {#AppVersion}
AppPublisher=Micrograph Maker
VersionInfoVersion={#AppVersion}
DefaultDirName={autopf}\Micrograph Maker
DefaultGroupName=Micrograph Maker
DisableProgramGroupPage=yes
DisableDirPage=auto
PrivilegesRequired=lowest
OutputDir=..\installer
OutputBaseFilename=MicrographMaker-{#AppVersion}-windows-setup
SetupIconFile=icons\MicrographMaker.ico
UninstallDisplayIcon={app}\MicrographMaker.exe
UninstallDisplayName=Micrograph Maker
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; 古い版の部品を消してから入れる（読み取りの記憶は %APPDATA%\edsfig にあるので消えない）
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "..\dist\MicrographMaker\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Micrograph Maker"; Filename: "{app}\MicrographMaker.exe"
Name: "{autodesktop}\Micrograph Maker"; Filename: "{app}\MicrographMaker.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\MicrographMaker.exe"; Description: "{cm:LaunchProgram,Micrograph Maker}"; Flags: nowait postinstall skipifsilent
