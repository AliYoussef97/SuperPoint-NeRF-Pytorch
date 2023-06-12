@echo off

REM %~p0 returns the path without the drive. 
REM %~d0 returns the drive.
cd /D %~dp0
cd..

echo Downloading ffmepg
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://github.com/GyanD/codexffmpeg/releases/download/2023-06-08-git-024c30aa3b/ffmpeg-2023-06-08-git-024c30aa3b-essentials_build.zip', 'ffmpeg.zip')"

echo Extracting ffmpeg zip file.
powershell -Command "Expand-Archive -Force ffmpeg.zip -DestinationPath .\utils\ffmpeg"


echo Deleting ffmpeg zip file.
del /f ffmpeg.zip

exit /b