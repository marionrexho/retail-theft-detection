@echo off
rem One-time setup on Windows. Safe to run again.
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Python is not installed. Install it with:
    echo     winget install Python.Python.3.12
    echo then close this window and run setup_windows.bat again.
    pause
    exit /b 1
)

rem The AI libraries support Python 3.10 to 3.12. Use the newest of those you have.
set "PYV="
for %%v in (3.12 3.11 3.10) do (
    if not defined PYV (
        py -%%v -c "import sys" >nul 2>nul && set "PYV=%%v"
    )
)
if not defined PYV (
    echo This project needs Python 3.10, 3.11 or 3.12 ^(newer versions are not supported
    echo by the AI libraries yet^). Your other Python versions are fine to keep.
    echo Install 3.12 with:
    echo     winget install Python.Python.3.12
    echo then close this window and run setup_windows.bat again.
    pause
    exit /b 1
)
echo Using Python %PYV%

if not exist "venv\Scripts\python.exe" (
    echo Creating the venv folder...
    py -%PYV% -m venv venv
    if errorlevel 1 (
        echo Could not create the venv folder.
        pause
        exit /b 1
    )
)

echo Installing packages. The first time takes a few minutes...
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed

echo.
echo Checking everything works...
venv\Scripts\python.exe -m pytest -q
if errorlevel 1 goto failed

echo.
echo Setup done. Open a command window in this folder and run:
echo     venv\Scripts\activate
echo Then see "Using it" in README.md.
pause
exit /b 0

:failed
echo.
echo Something failed above. Copy the error and ask for help.
echo To start over, delete the venv folder and run setup_windows.bat again.
pause
exit /b 1
