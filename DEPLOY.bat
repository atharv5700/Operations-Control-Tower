@echo off
REM One-click Docker build and run script for Windows

echo ========================================
echo AI Supply Chain Control Tower
echo Docker Deployment Script
echo ========================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Docker is not installed!
    echo.
    echo Please install Docker Desktop from:
    echo https://desktop.docker.com/win/main/amd64/Docker%%20Desktop%%20Installer.exe
    echo.
    echo After installation, run this script again.
    pause
    exit /b 1
)

echo [1/4] Docker detected: OK
docker --version

echo.
echo [2/4] Building Docker image...
echo This may take 10-15 minutes on first run.
docker build -t ai-supply-chain-tower .

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Docker build failed!
    echo Check the error messages above.
    pause
    exit /b 1
)

echo.
echo [3/4] Stopping any existing container...
docker stop supply-chain-tower >nul 2>&1
docker rm supply-chain-tower >nul 2>&1

echo.
echo [4/4] Starting container...
docker run -d -p 8501:8501 --name supply-chain-tower ai-supply-chain-tower

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to start container!
    pause
    exit /b 1
)

echo.
echo ========================================
echo SUCCESS! App is running in Docker!
echo ========================================
echo.
echo Access the app at:
echo   http://localhost:8501
echo.
echo Waiting 10 seconds for app to initialize...
timeout /t 10 /nobreak >nul

echo.
echo Opening browser...
start http://localhost:8501

echo.
echo Container commands:
echo   Stop:    docker stop supply-chain-tower
echo   Start:   docker start supply-chain-tower
echo   Logs:    docker logs supply-chain-tower
echo   Status:  docker ps
echo.
pause
