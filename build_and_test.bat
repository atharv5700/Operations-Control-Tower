@echo off
REM build_and_test.bat - Windows version of build and test script

echo Building AI Supply Chain Control Tower Docker Image...

docker build -t ai-supply-chain-tower:latest .

if %ERRORLEVEL% EQU 0 (
    echo Docker image built successfully!
    
    echo.
    echo Image size:
    docker images ai-supply-chain-tower:latest
    
    echo.
    echo Running quick test...
    
    REM Run container
    docker run -d --name supply-chain-test -p 8501:8501 ai-supply-chain-tower:latest
    
    echo Waiting for app to start (30 seconds)...
    timeout /t 30 /nobreak > nul
    
    REM Check if app is responding (Windows doesn't have curl by default, so we'll just show instructions)
    echo.
    echo App should be running now!
    echo Open browser to: http://localhost:8501
    echo.
    echo To stop the test container, run:
    echo   docker stop supply-chain-test
    echo   docker rm supply-chain-test
    
) else (
    echo Docker build failed!
    exit /b 1
)
