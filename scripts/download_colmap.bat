@echo off
setlocal enabledelayedexpansion

REM %~p0 returns the path without the drive. 
REM %~d0 returns the drive.
cd /D %~dp0
cd..
REM Check if CUDA is available
nvcc --version > nul 2>&1

if %errorlevel% EQU 0 (
    
    echo CUDA is available, Downloading Colmap CUDA
    powershell -Command "(New-Object Net.WebClient).DownloadFile('https://github.com/colmap/colmap/releases/download/3.8/COLMAP-3.8-windows-cuda.zip', 'COLMAP-3.8-windows-cuda.zip')"

    echo Extracting Colmap zip file.
    powershell -Command "Expand-Archive -Force COLMAP-3.8-windows-cuda.zip -DestinationPath .\utils\colmap"

    echo Deleting Colmap zip file.
    del /f COLMAP-3.8-windows-cuda.zip

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
    echo !USERPATH! | findstr /i "%CD%\\utils\\colmap\\COLMAP-3.8-windows-cuda\\" > nul

    if not errorlevel 1 (
        REM Path already exists
        echo Path already exists
        
    ) else (
        
        REM Append the new path to the existing PATH
        set "newPath=%CD%\utils\colmap\COLMAP-3.8-windows-cuda\;"
        set "updatedPath=!USERPATH!!newPath!"
        reg add "HKEY_CURRENT_USER\Environment" /v PATH /t !pathType! /d "!updatedPath!" /f

    )

 
) else (
    
    echo CUDA is not available, Downloading Colmap NO CUDA
    powershell -Command "(New-Object Net.WebClient).DownloadFile('https://github.com/colmap/colmap/releases/download/3.8/COLMAP-3.8-windows-no-cuda.zip', 'COLMAP-3.8-windows-no-cuda.zip')"

    echo Extracting Colmap zip file.
    powershell -Command "Expand-Archive -Force COLMAP-3.8-windows-no-cuda.zip -DestinationPath .\utils\colmap"

    echo Deleting Colmap zip file.
    del /f COLMAP-3.8-windows-no-cuda.zip

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
    echo !USERPATH! | findstr /i "%CD%\\utils\\colmap\\COLMAP-3.8-windows-no-cuda\\" > nul

    if not errorlevel 1 (
        REM Path already exists
        echo Path already exists
        
    ) else (
        
        REM Append the new path to the existing PATH
        set "newPath=%CD%\utils\colmap\COLMAP-3.8-windows-no-cuda\;"
        set "updatedPath=!USERPATH!!newPath!"

        reg add "HKEY_CURRENT_USER\Environment" /v PATH /t !pathType! /d "!updatedPath!" /f
    )

)

endlocal
exit /b
