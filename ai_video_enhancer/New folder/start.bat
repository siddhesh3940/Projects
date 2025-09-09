@echo off
echo Starting AI Video Enhancer...

echo.
echo Testing backend...
cd backend
python test_basic.py
if errorlevel 1 (
    echo.
    echo Installing basic requirements...
    pip install -r requirements_basic.txt
)

echo.
echo Starting Backend Server...
start cmd /k "python run.py"

cd ..
timeout /t 3 /nobreak > nul

echo.
echo Starting Frontend Server...
start cmd /k "npm run dev"

echo.
echo Both servers are starting...
echo Frontend: http://localhost:3000
echo Backend: http://localhost:8000
echo.
echo Note: AI models are optional. Basic enhancement will work without them.
pause