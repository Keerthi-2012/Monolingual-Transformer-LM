@echo off
REM Starts the Hindi LM demo. Uses HINDI_DEMO_PY if set, else the demo virtual environment if present, else "python".
REM Expects ckpt_best.pt in this folder (or pass --ckpt <path>). Then open http://127.0.0.1:7860
set "PY=%HINDI_DEMO_PY%"
if "%PY%"=="" if exist "D:\Keerthi\Personal\lm_demo_venv\Scripts\python.exe" set "PY=D:\Keerthi\Personal\lm_demo_venv\Scripts\python.exe"
if "%PY%"=="" set "PY=python"
"%PY%" -X utf8 "%~dp0app.py" %*
pause
