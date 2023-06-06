@echo off

REM %~p0 returns the path without the drive. 
REM %~d0 returns the drive.
cd %~dp0

call "%HOMEDRIVE%%HOMEPATH%\Anaconda3\scripts\activate.bat"

call conda create --name nerfstudio_cool -y python=3.8

call conda activate nerfstudio_cool

call conda install -y git 

python -m pip install --upgrade pip

pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

cd.. 

git clone https://github.com/nerfstudio-project/nerfstudio.git

cd nerfstudio

pip install --upgrade pip setuptools

pip install -e .

exit /b