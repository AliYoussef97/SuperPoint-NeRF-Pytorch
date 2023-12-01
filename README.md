  # Improving Relative Pose Estimation of Learning-based Feature Point Detectors and Descriptors using Novel-View Synthesis

  The project aims to enhance [SuperPoint](https://arxiv.org/abs/1712.07629) relative pose estimation, by incorporating techniques such as [NeRF](https://arxiv.org/abs/2003.08934)-based view generation and perspective projection geometry.
  Supervised by: Dr. Francisco Porto Guerra E Vasconcelos


  ## 1. Installation ##

  ### 1.1 Create Conda Environment ###

  ```
    conda create -n SP-Nerf -y python=3.8
  ```

  ### 1.2 Dependencies ###

  This project uses Ffmpeg, Colmap and [NerfStudio](https://github.com/nerfstudio-project/nerfstudio), run the following commands, to download the dependencies.

  ```
    ../scripts/download_nerfstudio.bat
    ../scripts/download_ffmpeg.bat
    ../scripts/download_colmap.bat
  ```

  **Note:**
  
  **1. NerfStudio uses CUDA 11.7 or 11.8, this project is built using CUDA 11.7. If you are using an older/newer version of CUDA, the following is the [CUDA Toolkit 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive) needed.**
  
  **2. The download_colmap and download_ffmpeg batch files add colmap and ffmpeg to the User Environment Variable Path. 
  After running the batch files, please close and reopen the terminal to account for the updated Environment Variables**


  ## 2. Running Colmap ##

  To run Colmap on a set of images or a video, follow these steps:
   1. Create a folder.
   2. For images:
      - Inside the folder, create a sub-folder called "images" and places images into the sub-folder.
  3. For a video:
      - Place the video file directly into the folder.


Run the following command run the colmap script.

  ```
    python ..\scripts\colmap.py --data-path {path to images folder or video file} --input-type {images or video} 
  ```

  e.g.
  ```
    python ..\scripts\colmap.py --data-path .\data\Video.MOV --input-type video
  ```
  

To display all available options run:
  ```
  python ..\scripts\colmap.py -h
  ```
  
  ## 3. NerfStudio ##

  Please refer to the NerfStudio [documentation](https://docs.nerf.studio/en/latest/index.html) to train a NeRF model using your data.

 It is recommended to save the model in the same location as your data for convenience and ease of use. This can be achieved by specifying the `-output-dir` argument in NerfStudio to the folder where your data is located. 

  ## 4. SuperPoint ##
  The SuperPoint implementation is based on:

  1. The [SuperPoint](https://arxiv.org/abs/1712.07629) paper.
  2. [Rémi Pautrat and Paul-Edouard Sarlin](https://github.com/rpautrat/SuperPoint) tensorflow implementation.


  ### 4.1 Setup ###

  
  In order to install the requirements and setup superpoint and the paths, run:
 
  ```
  make install
  ```

  You will be required to provide three different paths:

    1. Data_PATH: The path to the folder which will contain the datasets.
    2. CKPT_PATH: The path where the model's checkpoints are saved.
    3. EXPER_PATH: The path where the pseudo_labels are saved and any other experiments like HPatches etc.


  When MagicPoint is trained for the first time on Synthetic Shapes, the Synthetic Shapes dataset will be generated automatically and saved at the provided Data_PATH.

  The folder containing the datasets should be structured as follows:
```
datasets
|-- COCO
|  |-- images
|  |   |-- training
|  |   |-- validation
|-- HPatches
|   |-- i_ajustment
|   |   |--1.ppm
|   |   |--...
|   |   |--H_1_2
|   |-- ...
|-- synthetic_shapes
```
  
  ### 4.2 Usage ###

To display all available training options run:
  ```
  python engine.py -h
  ```
**Note:** This project has extra modifications to accommodate certain requirements. However, if you are only interested in the original paper or the tensorflow implementation, make sure the following is set for all steps:
```
During training:
1. --training.nerf_loss False
2. --training.training_nerf False

During Exporting pseudo_labels:
1. --pseudo_labels.enable_Homography_Adaptation True
```



### 4.3 Training MagicPoint on Synthetic Shapes ###

Before training set the following as True to follow the tensorflow implementation or False to follow the Original Paper.

```
--training.include_mask_loss True/False
```


To train Magicpoint on Synthetic Shapes, run:

```
python engine.py --config-path .\configs\magicpoint_syn.yaml --task train --training.include_mask_loss False
```

The checkpoint will be saved at the specified CHPT_PATH.

  ### 4.4 Exporting Pseudo Labels ###
  To export Pseudo Labels run:
  
  ```
  python engine.py --config_path .\configs\magicpoint_coco_export.yaml --task export_pseudo_labels
  ```

  The Pseudo Labels will be saved at .\experiments\outputs\
  
   For additional options during exporting, refer to section 4.1

### 4.5 Training MagicPoint on COCO Dataset ###

To train Magicpoint on COCO dataset, run:

```
python engine.py --config-path .\configs\magicpoint_coco_train.yaml --task train --training.validate_training False --include_mask_loss False
```

If Pseudo Labels were not exported for the COCO validation images, you can set the following to avoid validating during training.
```
--training.validate_training False
```

Steps 4.4 and 4.5 should be repeated once or twice to improve the generalization ability of the base MagicPoint architecture on real images.

### 4.6 Evaluating HPatches Detector Repeatability ###

Run the following to export HPatches data used to evaluate the detector repeatability:

```
python engine.py --config_path .\configs\magicpoint_repeatability.yaml --task export_HPatches_Repeatability
```

In the configuration file, change the `alteration` argument as `v` to export data on the HPatches viewpoint dataset or `i` to export the data on the HPatches illumination dataset.

The data will be saved in .\EXPER_PATH\reapeatability\

To evaluate the repeatability use the detector_repeatability_hpatches.ipynb notebook.

### 4.7 Training SuperPoint on COCO Dataset ###


  **Note:** In order to follow the Original SuperPoint paper during SuperPoint training, set the `descriptor_normalization` variable as False in the configuration file. If set as True, it follows the tensorflow implementation. 

Run the following command to train the SuperPoint model on the COCO dataset.

Again, if Pseudo Labels were not generated for the validation images, you can set validate_training argument as False.

```
python engine.py --config-path .\configs\superpoint_coco_train.yaml --task train --training.validate_training False --include_mask_loss False
```

### 4.8 Evaluating HPatches Homography Estimation ###

Run the following to export HPatches data used to evaluate Homography Estimation:

```
python engine.py --config_path .\configs\superpoint_descriptors.yaml --task export_HPatches_Descriptors
```

In the configuration file, change the `alteration` argument as `v` to export data on the HPatches viewpoint dataset or `i` to export the data on the HPatches illumination dataset, or `all` to export data for both viewpoint and illumination combined.

The data will be saved in .\EXPER_PATH\descriptors\

To evaluate the repeatability use the descriptors_evaluation_hpatches.ipynb notebook.


## Credits

The implementation was developed by [Ali Youssef](https://github.com/AliYoussef97). Special thanks to the authors of [SuperPoint](https://arxiv.org/abs/1712.07629) and [Rémi Pautrat and Paul-Edouard Sarlin](https://github.com/rpautrat/SuperPoint) for their tensorflow implementation which a lot of work was based on.








