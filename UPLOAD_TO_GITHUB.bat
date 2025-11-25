@echo off
REM ================================================
REM GitHub Upload Script for Windows
REM ================================================

echo.
echo ========================================
echo  GitHub Upload - AI Supply Chain
echo ========================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Git is not installed!
    echo.
    echo Please install Git from: https://git-scm.com/download/win
    echo.
    pause
    exit /b 1
)

echo Step 1: Git detected
echo.

REM Check if already initialized
if exist ".git" (
    echo Git already initialized in this directory.
    echo.
) else (
    echo Initializing git repository...
    git init
    echo.
)

echo Step 2: Adding all files...
git add .
echo.

echo Step 3: Creating commit...
git commit -m "🎉 Production-ready AI Supply Chain Control Tower with ML forecasting"
echo.

echo Step 4: Setting up GitHub remote...
echo.
echo Now you need to:
echo 1. Go to https://github.com/new
echo 2. Create a new repository named: AI-Supply-Chain-Control-Tower
echo 3. DON'T initialize with README
echo 4. Copy the HTTPS URL (like: https://github.com/USERNAME/AI-Supply-Chain-Control-Tower.git)
echo.
set /p GITHUB_URL="Paste your GitHub repo URL here: "

echo.
echo Connecting to GitHub...
git remote remove origin 2>nul
git remote add origin %GITHUB_URL%
git branch -M main

echo.
echo Step 5: Pushing to GitHub...
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo  SUCCESS! Code uploaded to GitHub!
    echo ========================================
    echo.
    echo Your repository: %GITHUB_URL%
    echo.
    echo Next steps:
    echo 1. Go to https://streamlit.io/cloud
    echo 2. Sign up with GitHub
    echo 3. Deploy this repository
    echo 4. Get your live demo URL
    echo.
    echo See GITHUB_SETUP.md for full instructions!
    echo.
) else (
    echo.
    echo ERROR: Push failed!
    echo.
    echo Possible issues:
    echo - Wrong repository URL
    echo - No permission to push
    echo - Need to authenticate
    echo.
    echo Try:
    echo git push -u origin main
    echo.
)

pause
