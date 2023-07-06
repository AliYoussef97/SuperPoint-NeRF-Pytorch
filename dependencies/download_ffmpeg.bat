@echo off
setlocal enabledelayedexpansion

REM %~p0 returns the path without the drive.
REM %~d0 returns the drive.
cd /D %~dp0
cd..

echo Downloading ffmepg
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://github.com/GyanD/codexffmpeg/releases/download/2023-06-08-git-024c30aa3b/ffmpeg-2023-06-08-git-024c30aa3b-essentials_build.zip', 'ffmpeg.zip')"

echo Extracting ffmpeg zip file.
powershell -Command "Expand-Archive -Force ffmpeg.zip -DestinationPath .\utils\ffmpeg"

ren ".\utils\ffmpeg\ffmpeg-2023-06-08-git-024c30aa3b-essentials_build" "ffmpeg"

echo Deleting ffmpeg zip file.
del /f ffmpeg.zip

REM Get the current value of the Path variable
for /f "usebackq tokens=2,*" %%A in (`reg query "HKEY_CURRENT_USER\Environment" /v PATH`) do set USERPATH=%%B

for /f "tokens=2,*" %%A in ('reg query "HKEY_CURRENT_USER\Environment" /v PATH') do (
    if /i "%%A"=="REG_SZ" (
        set "pathType=REG_SZ"
    ) else if /i "%%A"=="REG_EXPAND_SZ" (
        set "pathType=REG_EXPAND_SZ"
    )
)

REM Check if the path already exists in the variable
echo !USERPATH! | findstr /i "%CD%\\utils\\ffmpeg\\ffmpeg\\bin\\" > nul

if not errorlevel 1 (
    REM Path already exists
    echo Path already exists
    
) else (
    
    REM Append the new path to the existing PATH
    set "newPath=%CD%\utils\ffmpeg\ffmpeg\bin\;"
    set "updatedPath=!USERPATH!!newPath!"
    
    echo Adding ffmpeg to Environment Path
    reg add "HKEY_CURRENT_USER\Environment" /v PATH /t !pathType! /d "!updatedPath!" /f
)

endlocal

exit /b