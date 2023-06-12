#!/usr/bin/env python3

import os
from glob import glob
import subprocess
import shutil
from typing import Literal
import tyro
import dataclasses
import sys
import textwrap

cwd = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(cwd)


@dataclasses.dataclass
class colmap_args:
    camera_model: Literal["OPENCV", "RADIAL", "SIMPLE_RADIAL"] = "OPENCV"
    """Camera model to use for feature extraction."""
    colmap_matcher: Literal["vocab_tree", "sequential", "exhaustive", "spatial", "transitive"] = "vocab_tree"
    """Colmap matching method to use."""
    nerfstudio_transforms: bool = True 
    """Run nerfstudio on colmap's output to find transforms.json"""


@dataclasses.dataclass
class ffmpeg_args:
    images_out_format: Literal["jpg","png"] = "jpg"
    """Choose output format of extracted images from video, either jpg or png."""
    fps: int = 1
    """Frames per second"""


class main():
    """main class, script backbone.
    
    Args:
        data_path: Path to data.
        input_type: Input type, choose images or video.
        gpu: Choose between gpu or no-gpu.
    """

    def __init__(self,
                 data_path: os.path.realpath,
                 input_type: Literal["images","video"],
                 colmap: colmap_args,
                 ffmpeg: ffmpeg_args,
                 gpu: bool = True) -> None:
        
        self.data_path = data_path
        self.input_type = input_type
        self.gpu = int(gpu)
        self.colmap_args = colmap
        self.ffmpeg_args = ffmpeg

        self.check_dependencies()

        if self.input_type == "video":
            self.run_ffmpeg()
        
        self.run_colmap()

    def check_dependencies(self) -> None:

        ffmpeg_path = os.path.join(ROOT_DIR, '**', "ffmpeg.exe")
        ffmpeg = glob(ffmpeg_path, recursive=True)

        colmap_path = os.path.join(ROOT_DIR, '**', "colmap.bat")
        colmap = glob(colmap_path, recursive=True)

        if not ffmpeg:
            message = "Please run the download_ffmpeg batch file and open a new terminal to account for updated environment path"
            wrapped_message = textwrap.fill(message, width=40)
            print(f"\nERROR\n{'=' * 40}\n{wrapped_message}\n{'=' * 40}\n")          
            sys.exit(1)
        
        if not colmap:
            message = "Please run the download_colmap batch file and open a new terminal to account for updated environment path"
            wrapped_message = textwrap.fill(message, width=40)
            print(f"\nERROR\n{'=' * 40}\n{wrapped_message}\n{'=' * 40}\n")          
            sys.exit(1)

    def run_ffmpeg(self) -> None:

        ffmpeg_path = os.path.join(ROOT_DIR, '**', "ffmpeg.exe")
        ffmpeg = glob(ffmpeg_path, recursive=True)
        ffmpeg = ffmpeg[0]
        
        dataset_folder_path = os.path.dirname(self.data_path)
        images_path = os.path.realpath(os.path.join(dataset_folder_path,"images"))
        
        if os.path.exists(images_path):
            shutil.rmtree(images_path)
        os.system(f"mkdir {images_path}")

        if self.ffmpeg_args.images_out_format ==".jpg":
            ffmpeg_cmd = f"{ffmpeg} -i {self.data_path} -q:v 1 -qmin 1 -qmax 1 -vf \"fps={self.fps}\" {images_path}\image%05d.jpg"
        else:
            ffmpeg_cmd = f"{ffmpeg} -i {self.data_path} -vf \"fps={self.ffmpeg_args.fps}\"  {images_path}\image%05d.png"
        
        os.system(ffmpeg_cmd)


    def run_colmap(self) -> None:
    
        colmap_path = os.path.join(ROOT_DIR, '**', "colmap.bat")
        colmap = glob(colmap_path, recursive=True)
        colmap = colmap[0]

        dataset_folder_path = os.path.dirname(self.data_path)
        images_path = os.path.realpath(os.path.join(dataset_folder_path,"images"))
        output_path = os.path.realpath(os.path.join(dataset_folder_path,f"{os.path.basename(dataset_folder_path)}_data_output"))

        os.system(f"mkdir {output_path}")
        
        # Feature extraction
        db_path = os.path.realpath(output_path+"\database.db")
        feature_extraction_cmd = f"{colmap} feature_extractor --ImageReader.camera_model {self.colmap_args.camera_model} --ImageReader.single_camera 1 --SiftExtraction.use_gpu {self.gpu} --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1 --image_path {images_path} --database_path {db_path} "
        os.system(feature_extraction_cmd)


        # Feature Matching
        feature_matching_cmd = f"{colmap} {self.colmap_args.colmap_matcher}_matcher --SiftMatching.use_gpu {self.gpu} --SiftMatching.guided_matching 1 --database_path {db_path}"

        if self.colmap_args.colmap_matcher in ["vocab_tree","sequential"]:
            vocab_path = os.path.realpath(os.path.join(output_path, "vocab_tree_flickr100K_words32K.bin"))
            if not glob(vocab_path):
                vocab_cmd = ["powershell","-Command",
                        "(New-Object Net.WebClient).DownloadFile('https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin', '{}\\vocab_tree_flickr100K_words32K.bin')".format(output_path)]  
                subprocess.run(vocab_cmd, shell = True)
            matcher_vocab = "VocabTree" if self.colmap_args.colmap_matcher == "vocab_tree" else "Sequential"
            feature_matching_cmd += f" --{matcher_vocab}Matching.vocab_tree_path {vocab_path}"
        os.system(feature_matching_cmd)
        
        # Mapping
        colmap_mapping_path = os.path.realpath(os.path.join(output_path, "colmap", "sparse"))
        os.system(f"Mkdir {colmap_mapping_path}")
        mapper_cmd = f"{colmap} mapper --Mapper.ba_local_max_num_iterations 50 --Mapper.ba_local_max_refinements 3 --Mapper.ba_global_max_num_iterations 100 --database_path {db_path} --image_path {images_path} --output_path {colmap_mapping_path}"
        os.system(mapper_cmd)


        # Bundle Adjustment (To further refine the mapping).
        BA_path = os.path.realpath(os.path.join(colmap_mapping_path,"0"))
        os.system(f"Mkdir {BA_path}")
        BA_cmd = f"{colmap} bundle_adjuster --BundleAdjustment.refine_principal_point 1 --input_path {BA_path} --output_path {BA_path}"
        os.system(BA_cmd)


        # Create transforms.json file using nerfstudio.
        if self.colmap_args.nerfstudio_transforms:
            NS_cmd = f"ns-process-data images --data {images_path} --output-dir {output_path} --skip-colmap --colmap-model-path {BA_path}"
            if self.gpu:
                NS_cmd += " --gpu"
            os.system(NS_cmd)

    

if __name__ == "__main__":
   tyro.cli(main)