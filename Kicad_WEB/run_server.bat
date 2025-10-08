@echo off
setlocal

REM Always run with the local venv Python
set PY=.\.venv\Scripts\python.exe

REM Optional UI origin for your Vite app (tweak as needed)
set VITE_ORIGIN=http://localhost:5174

REM Start FastAPI
REM %PY% -m uvicorn companion.app:APP --reload --host 127.0.0.1 --port 8000
%PY% -m uvicorn companion.app2:APP --reload --host 127.0.0.1 --port 8010
endlocal
