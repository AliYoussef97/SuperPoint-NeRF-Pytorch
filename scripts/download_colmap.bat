@echo off

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
    powershell -Command "Expand-Archive -Force COLMAP-3.8-windows-cuda.zip -DestinationPath .\colmap"

    echo Deleting Colmap zip file.
    del /f COLMAP-3.8-windows-cuda.zip

 
) else (
    
    echo CUDA is not available, Downloading Colmap NO CUDA
    powershell -Command "(New-Object Net.WebClient).DownloadFile('https://github.com/colmap/colmap/releases/download/3.8/COLMAP-3.8-windows-no-cuda.zip', 'COLMAP-3.8-windows-no-cuda.zip')"

    echo Extracting Colmap zip file.
    powershell -Command "Expand-Archive -Force COLMAP-3.8-windows-no-cuda.zip -DestinationPath .\colmap"

    echo Deleting Colmap zip file.
    del /f COLMAP-3.8-windows-no-cuda.zip

)

exit /b