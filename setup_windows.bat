@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if errorlevel 1 goto missing_python

py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
if errorlevel 1 goto missing_python

if exist ".venv" (
    set "VENV_OK="
    if exist ".venv\Scripts\python.exe" (
        ".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
        if not errorlevel 1 set "VENV_OK=1"
    )
    if not defined VENV_OK (
        echo El entorno .venv existente no es utilizable con Python 3.12.
        choice /C YN /M "Confirmas eliminar solo la carpeta .venv y recrearla con Python 3.12"
        if errorlevel 2 goto cancelled
        rmdir /s /q ".venv"
        if exist ".venv" goto failed
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo Creando entorno virtual con Python 3.12...
    py -3.12 -m venv .venv
    if errorlevel 1 goto failed
)

echo Actualizando pip dentro de .venv...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed

echo Instalando dependencias...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed

echo Comprobando dependencias...
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed

echo Comprobando importaciones principales...
".venv\Scripts\python.exe" -c "import streamlit, pandas, numpy, sklearn, scipy, matplotlib, openpyxl, xlrd, PIL; from sklearn.neural_network import MLPClassifier; print('Importaciones correctas; Python', __import__('sys').version.split()[0], '| Streamlit', streamlit.__version__, '| Pillow', PIL.__version__)"
if errorlevel 1 goto failed

echo.
echo Instalacion verificada. Para iniciar StockSmart ML:
echo   .venv\Scripts\python.exe -m streamlit run app.py
exit /b 0

:missing_python
echo No se encontro un Python 3.12 accesible mediante el lanzador "py".
echo Instala Python 3.12.10 desde:
echo https://www.python.org/downloads/release/python-31210/
echo Luego vuelve a ejecutar setup_windows.bat. Este script no instala Python.
pause
exit /b 1

:cancelled
echo No se modifico .venv. Puedes conservarlo y revisar su configuracion manualmente.
pause
exit /b 1

:failed
echo La configuracion fallo. Revisa el mensaje anterior; no se modificaron datos ni modelos.
pause
exit /b 1
