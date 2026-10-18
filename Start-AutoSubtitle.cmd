@echo off
setlocal
set "AUTOSUB_MODEL_DIR=%~dp0models"
set "HF_XET_CACHE=%~dp0.cache\xet"
set "AUTOSUB_ENABLE_REPAIR_UI="
if exist "%~dp0dist\AutoSubtitle-GPU-0.9.6\AutoSubtitle-GPU-0.9.6.exe" (
    start "" /D "%~dp0" "%~dp0dist\AutoSubtitle-GPU-0.9.6\AutoSubtitle-GPU-0.9.6.exe"
    exit /b 0
)
if exist "%~dp0dist\AutoSubtitle-0.9.6\AutoSubtitle-0.9.6.exe" (
    start "" /D "%~dp0" "%~dp0dist\AutoSubtitle-0.9.6\AutoSubtitle-0.9.6.exe"
    exit /b 0
)
if exist "%~dp0dist\AutoSubtitle-GPU-0.9.5\AutoSubtitle-GPU-0.9.5.exe" (
    start "" /D "%~dp0" "%~dp0dist\AutoSubtitle-GPU-0.9.5\AutoSubtitle-GPU-0.9.5.exe"
    exit /b 0
)
if exist "%~dp0dist\AutoSubtitle-0.9.5\AutoSubtitle-0.9.5.exe" (
    start "" /D "%~dp0" "%~dp0dist\AutoSubtitle-0.9.5\AutoSubtitle-0.9.5.exe"
    exit /b 0
)
if exist "%~dp0dist\AutoSubtitle-GPU-0.9.4\AutoSubtitle-GPU-0.9.4.exe" (
    start "" /D "%~dp0" "%~dp0dist\AutoSubtitle-GPU-0.9.4\AutoSubtitle-GPU-0.9.4.exe"
    exit /b 0
)
if exist "%~dp0dist\AutoSubtitle-0.9.4\AutoSubtitle-0.9.4.exe" (
    start "" /D "%~dp0" "%~dp0dist\AutoSubtitle-0.9.4\AutoSubtitle-0.9.4.exe"
    exit /b 0
)
if exist "%~dp0dist\AutoSubtitle-GPU-0.9.3\AutoSubtitle-GPU-0.9.3.exe" (
    start "" /D "%~dp0" "%~dp0dist\AutoSubtitle-GPU-0.9.3\AutoSubtitle-GPU-0.9.3.exe"
    exit /b 0
)
if exist "%~dp0dist\AutoSubtitle-0.9.3\AutoSubtitle-0.9.3.exe" (
    start "" /D "%~dp0" "%~dp0dist\AutoSubtitle-0.9.3\AutoSubtitle-0.9.3.exe"
    exit /b 0
)
if exist "%~dp0dist\AutoSubtitle-GPU-0.9.2\AutoSubtitle-GPU-0.9.2.exe" (
    start "" /D "%~dp0" "%~dp0dist\AutoSubtitle-GPU-0.9.2\AutoSubtitle-GPU-0.9.2.exe"
    exit /b 0
)
if exist "%~dp0dist\AutoSubtitle-0.9.2\AutoSubtitle-0.9.2.exe" (
    start "" /D "%~dp0" "%~dp0dist\AutoSubtitle-0.9.2\AutoSubtitle-0.9.2.exe"
    exit /b 0
)
if exist "%~dp0dist\AutoSubtitle-GPU-0.9.1\AutoSubtitle-GPU-0.9.1.exe" (
    start "" /D "%~dp0" "%~dp0dist\AutoSubtitle-GPU-0.9.1\AutoSubtitle-GPU-0.9.1.exe"
    exit /b 0
)
if exist "%~dp0dist\AutoSubtitle-0.9.1\AutoSubtitle-0.9.1.exe" (
    start "" /D "%~dp0" "%~dp0dist\AutoSubtitle-0.9.1\AutoSubtitle-0.9.1.exe"
    exit /b 0
)
if exist "%~dp0dist\AutoSubtitle-GPU\AutoSubtitle-GPU.exe" (
    start "" /D "%~dp0" "%~dp0dist\AutoSubtitle-GPU\AutoSubtitle-GPU.exe"
    exit /b 0
)
if exist "%~dp0dist\AutoSubtitle\AutoSubtitle.exe" (
    start "" /D "%~dp0" "%~dp0dist\AutoSubtitle\AutoSubtitle.exe"
    exit /b 0
)
if exist "%~dp0.venv\Scripts\pythonw.exe" (
    start "" /D "%~dp0" "%~dp0.venv\Scripts\pythonw.exe" -m autosub
    exit /b 0
)
echo Please install the development environment described in README.md.
pause
exit /b 1
