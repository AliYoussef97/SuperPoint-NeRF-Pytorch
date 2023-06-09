  # Improving Visual Landmark Detection and Matching for Video-based Mapping using NeRF-based View Generation and Self-Supervised Machine Learning

  The project aims to enhance [SuperPoint](https://arxiv.org/abs/1712.07629), a real-time feature point detection and description tool, by incorporating advanced techniques such as [NeRF](https://arxiv.org/abs/2003.08934)-based view generation and self-supervised machine learning. The proposed approach focuses on improving the model's geometric consistency and overall performance. 

  Supervised by: Dr. Francisco Porto Guerra E Vasconcelos

  ***This project is currently a work in progress***

  ## 1. Installation ##

  ### 1.1 Create Conda Environment ###

  ```
    conda create -n UCL-Project -y python=3.8
  ```

  ### 1.2 Dependencies ###

  This project uses [NerfStudio](https://github.com/nerfstudio-project/nerfstudio), run the following command, to install nerfstudio and it's dependencies.

  ```
    ../scripts/download_nerfstudio.bat
  ```

  **Note: NerfStudio uses CUDA 11.7 or 11.8, this project is built using CUDA 11.7. If you are using an older/newer version of CUDA, the following is the [CUDA Toolkit 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive) to needed.**


  ## 2. Running Colmap ##

  To run Colmap on a set of images or a video, follow these steps:
   1. Create a folder.
   2. For images:
      - Inside the folder, create a sub-folder called "images" and places images into the sub-folder.
  3. For a video:
      - Place the video file directly into the folder.


Run the following command run the colmap script.

  ```
    pyhton ..\scripts\colmap.py {images or video} --data_path {path to images folder or video file}
  ```

  e.g.
  ```
    pyhton ..\scripts\colmap.py video --data_path \data\Video.MOV
  ```
  or

  ```
    pyhton ..\scripts\colmap.py images --data_path \data\images
  ```
  
  ## 3. NerfStudio ##

  Please refer to the NerfStudio [documentation](https://docs.nerf.studio/en/latest/index.html) to train a NeRF model using your data.

 It is recommended to save the model in the same location as your data for convenience and ease of use. This can be achieved by specifying the `-output-dir` argument in NerfStudio to the folder where your data is located. 


  For the full Colamp options, use the `help` command, as follows:

  ```
    pyhton ../scripts/colmap.py --help
  ```


