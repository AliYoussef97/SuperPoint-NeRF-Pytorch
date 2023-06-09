#!/usr/bin/env python3

import os
from glob import glob
import argparse
import subprocess
import shutil


cwd = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(cwd)


def run_ffmpeg(args):

    ffmpeg_path = os.path.join(ROOT_DIR, '**', "ffmpeg.exe")
    ffmpeg = glob(ffmpeg_path, recursive=True)
    
    # if ffmpeg = []
    if not ffmpeg:
        # Download ffmpeg then set ffmpeg as [path\ffmpeg.exe]
        os.system(os.path.join(cwd,"download_ffmpeg.bat"))
        ffmpeg = glob(ffmpeg_path, recursive= True)
    
    # if ffmpeg = [path\ffmpeg.exe]
    if ffmpeg:
        # set ffmpeg = path\ffmpeg.exe
        ffmpeg = ffmpeg[0]
    
    dataset_folder_path = os.path.dirname(args.data_path)
    images_path = os.path.realpath(os.path.join(dataset_folder_path,"images"))
    
    if os.path.exists(images_path):
        shutil.rmtree(images_path)
    os.system(f"mkdir {images_path}")

    if args.video_out_format ==".jpg":
        ffmpeg_cmd = f"{ffmpeg} -i {args.data_path} -q:v 1 -qmin 1 -qmax 1 -vf \"fps={args.fps}\" {images_path}\image%05d.jpg"
    else:
        ffmpeg_cmd = f"{ffmpeg} -i {args.data_path} -vf \"fps={args.fps}\"  {images_path}\image%05d.png"
    
    os.system(ffmpeg_cmd)


def run_colmap(args):
    
    colmap_path = os.path.join(ROOT_DIR, '**', "colmap.bat")
    colmap = glob(colmap_path, recursive=True)
    
    # if colmap = []
    if not colmap:
        # Download colmap then set colmap as [path\colmap.bat]
        os.system(os.path.join(cwd,"download_colmap.bat"))
        colmap = glob(colmap_path, recursive= True)
    
    # if colmap = [path\colmap.bat]
    if colmap:
        # set colmap = path\colmap.bat
        colmap = colmap[0]

    dataset_folder_path = os.path.dirname(args.data_path)
    images_path = os.path.realpath(os.path.join(dataset_folder_path,"images"))
    output_path = os.path.realpath(os.path.join(dataset_folder_path,f"{os.path.basename(dataset_folder_path)}_data_output"))

    os.system(f"mkdir {output_path}")
    
    # Feature extraction
    db_path = os.path.realpath(output_path+"\database.db")
    feature_extraction_cmd = f"{colmap} feature_extractor --ImageReader.camera_model {args.camera_model} --ImageReader.single_camera 1 --SiftExtraction.use_gpu {args.gpu} --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1 --image_path {images_path} --database_path {db_path} "
    os.system(feature_extraction_cmd)


    # Feature Matching
    feature_matching_cmd = f"{colmap} {args.colmap_matcher}_matcher --SiftMatching.use_gpu {args.gpu} --SiftMatching.guided_matching 1 --database_path {db_path}"

    if args.colmap_matcher in ["vocab_tree","sequential"]:
        vocab_path = os.path.realpath(os.path.join(output_path, "vocab_tree_flickr100K_words32K.bin"))
        if not glob(vocab_path):
            vocab_cmd = ["powershell","-Command",
                    "(New-Object Net.WebClient).DownloadFile('https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin', '{}\\vocab_tree_flickr100K_words32K.bin')".format(output_path)]  
            subprocess.run(vocab_cmd, shell = True)
        matcher_vocab = "VocabTree" if args.colmap_matcher == "vocab_tree" else "Sequential"
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
    if args.nerfstudio_transforms:
        NS_cmd = f"ns-process-data images --data {images_path} --output-dir {output_path} --skip-colmap --colmap-model-path {BA_path}"
        if args.gpu:
            NS_cmd += " --gpu"
        os.system(NS_cmd)
    
    
def parser():
    parser = argparse.ArgumentParser(description = 'Colmap parser')
    parser.add_argument("input_type", nargs='?', default="", choices=["images","video"], type = str,
                        help="Choose from input type choices, images or video.")
    parser.add_argument("--data_path", default="", type = str,
                        help='Path to images folder, or Video file')
    parser.add_argument("--fps", default=1, type = int,
                        help='Frames per second used for ffmpeg.')
    parser.add_argument("--video_out_format", default=".jpg",choices=[".jpg",".png"],type=str,
                        help="Choose output format of extracted images from video, either jog or png.")
    parser.add_argument("--camera_model", default="OPENCV", choices=["OPENCV", "RADIAL", "SIMPLE_RADIAL"], type=str,
                    help="(default: %(default)s). Camera model to use for feature extraction.")
    parser.add_argument("--gpu", type = int, default = 0,
                        help='(default: %(default)s). Set to 0 if the device is CPU, else set to 1 if CUDA is available.')
    parser.add_argument("--colmap_matcher", default = "vocab_tree", choices=["vocab_tree", "sequential", "exhaustive", "spatial", "transitive"], 
                        help="(default: %(default)s). Colmap matching method to use.")
    parser.add_argument("--nerfstudio_transforms", default = 1, type = int,
                        help = "(default: %(default)s). Run nerfstudio on colmap output to find transforms.json")
    
    return parser.parse_args()



if __name__ == "__main__":
    args = parser()

    if args.input_type == "video":
        run_ffmpeg(args)
    run_colmap(args)