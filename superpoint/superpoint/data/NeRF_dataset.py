import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.functional as TF
from superpoint.data.data_utils.kp_utils import warp_points_NeRF, compute_keypoint_map, filter_points
from superpoint.data.data_utils.photometric_augmentation import Photometric_aug
from superpoint.settings import DATA_PATH, EXPER_PATH


class NeRF(Dataset):
    def __init__(self, data_config, task = "training" ,device="cpu") -> None:
        super(NeRF, self).__init__()
        self.config = data_config
        self.device = device
        self.action = "training" if task == "training" else "validation" if task == "validation" else "test"
        self.samples = self._init_dataset()
        self.camera_intrinsic_matrix = self.get_camera_intrinsic(self.config["image_size"],self.config["fov"])
        
        if self.config["augmentation"]["photometric"]["enable"]:
            self.photometric_aug = Photometric_aug(self.config["augmentation"]["photometric"])


    def _init_dataset(self):
        """
        List of images' path and names to be processed.
        """
        data_dir = Path(DATA_PATH, "NeRF", self.config["data_dir"], "images", self.action)
        image_paths = sorted(list(data_dir.iterdir()))
        if self.config["truncate"]:
            image_paths = image_paths[:int(self.config["truncate"]*len(image_paths))]
        names = [p.stem for p in image_paths]
        image_paths = [str(p) for p in image_paths]
        files = {"image_paths":image_paths, "names":names}

        camera_transform_dir = Path(DATA_PATH, "NeRF", self.config["data_dir"], "camera_transforms", self.action)
        depth_dir = Path(DATA_PATH, "NeRF", self.config["data_dir"], "depth", self.action)
        camera_transform_paths = []
        depth_paths = []
        for n in files["names"]:
            p = Path(camera_transform_dir,'{}.npy'.format(n))
            camera_transform_paths.append(str(p))
            p = Path(depth_dir,'{}.npy'.format(n))
            depth_paths.append(str(p))
        files["camera_transform_paths"] = camera_transform_paths
        files["depth_paths"] = depth_paths

        if self.config["has_labels"]:
            # Load camera transformation matrix and depth map and labels.
            label_dir = Path(EXPER_PATH, self.config["has_labels"], self.action)
            label_paths = []
            for n in files["names"]:
                p = Path(label_dir,'{}.npy'.format(n))
                label_paths.append(str(p))
            files["label_paths"] = label_paths
            
        return files

    def __len__(self):
        return len(self.samples["image_paths"])

        
    def get_camera_intrinsic(self,shape,fov):
        '''
        Calculate camera intrinsic matrix.
        '''
        H, W = shape

        c_x = W//2
        c_y = H//2

        fov = np.deg2rad(fov)
        F_L = c_y/np.tan(fov/2)

        cam_intrinsic_matrix = np.array([ [ F_L,0,c_x] , 
                                          [ 0,F_L,c_y] , 
                                          [ 0, 0, 1  ] ],dtype=np.float32)
        
        cam_intrinsic_matrix = torch.as_tensor(cam_intrinsic_matrix, dtype=torch.float32, device=self.device)
        
        return cam_intrinsic_matrix
    

    def axis_transform(self,cam_matrix):
        '''
        Transform camera transformation matrix axis.
        '''
        reverse = np.diag([1, -1, -1, 1])
        cam_matrix =  cam_matrix @ reverse

        return cam_matrix
    

    def get_rotation_translation(self, transformation_matrix):
        # Get rotation and translation from camera transform
        rotation = transformation_matrix[:3, :3]
        rotation = torch.as_tensor(rotation, dtype=torch.float32,device=self.device)

        translation = transformation_matrix[:3, 3].reshape(3, 1)
        translation = torch.as_tensor(translation, dtype=torch.float32,device=self.device)
        return rotation, translation
    

    def random_frame(self,random_frame):
        
        data_len = len(self.samples["image_paths"])
        
        if random_frame == 0:
            frames = np.arange(random_frame+0.07*data_len,random_frame+0.15*data_len,1)
            return random.choice(frames)
        
        if random_frame == data_len-1:
            frames = np.arange(random_frame-0.15*data_len,random_frame-0.07*data_len,1)
            return random.choice(frames)
        
        if random_frame - 0.15*data_len < 0:
            frames = np.arange(random_frame+0.07*data_len,random_frame+0.15*data_len,1)
            return random.choice(frames)
        
        if random_frame + 0.15*data_len > data_len-1:
            frames = np.arange(random_frame-0.15*data_len,random_frame-0.07*data_len)
            return random.choice(frames)
        
        else:
            return random.choice(np.concatenate((np.arange(random_frame-0.15*data_len,random_frame-0.07*data_len,1),
                                                 np.arange(random_frame+0.07*data_len,random_frame+0.15*data_len,1)),
                                                 axis=0))
            

    def read_image(self, image):
        image = torchvision.io.read_file(image)
        image = torchvision.io.decode_image(image,torchvision.io.ImageReadMode.GRAY)
        return image.squeeze(0).to(torch.float32).to(self.device)
    

    def downsample_data(self, data):
        
        H_ds, W_ds = self.config["downsample_size"]
        
        i,j,h,w = torchvision.transforms.RandomCrop.get_params(data["raw"]["image"], output_size=(H_ds,W_ds))

        data["raw"]["image"] = data["raw"]["image"][i:i+h, j:j+w]
        data["warp"]["image"] = data["warp"]["image"][i:i+h, j:j+w]

        data["raw"]["valid_mask"] = data["raw"]["valid_mask"][i:i+h, j:j+w]
        data["warp"]["valid_mask"] = data["warp"]["valid_mask"][i:i+h, j:j+w]

        data["raw"]["kpts_heatmap"] = data["raw"]["kpts_heatmap"][i:i+h, j:j+w]
        data["warp"]["kpts_heatmap"] = data["warp"]["kpts_heatmap"][i:i+h, j:j+w]

        data["raw"]["kpts"] = torch.nonzero(data["raw"]["kpts_heatmap"], as_tuple=False)
        data["warp"]["kpts"] = torch.nonzero(data["warp"]["kpts_heatmap"], as_tuple=False)

        data["raw"]["input_depth"] = data["raw"]["input_depth"][i:i+h, j:j+w]

        data["camera_intrinsic_matrix"] = self.get_camera_intrinsic(self.config["downsample_size"],self.config["fov"])

        return data

    
    def __getitem__(self, index):

        image = self.samples["image_paths"][index]  
        image = self.read_image(image)
        
        input_name = self.samples["names"][index]
        

        data = {"raw":{'image':image},
                "name":input_name}
        input_transformation = np.load(self.samples["camera_transform_paths"][index])
        input_transformation = self.axis_transform(input_transformation)
        input_rotation, input_translation = self.get_rotation_translation(input_transformation)

        input_depth = np.load(self.samples["depth_paths"][index])
        input_depth = torch.as_tensor(input_depth, dtype=torch.float32, device=self.device)

        data["raw"]["input_depth"] = input_depth
        data["raw"]["input_rotation"] = input_rotation
        data["raw"]["input_translation"] = input_translation
        data["camera_intrinsic_matrix"] = self.camera_intrinsic_matrix
        
        # Add labels if exists.
        if self.config["has_labels"]: # Only for training/validaiton/test not exporting pseudo labels.

            points = self.samples["label_paths"][index]
            points = np.load(points)
            points = torch.as_tensor(points, dtype=torch.float32, device=self.device)
            data["raw"]["kpts"] = points
            data["raw"]["kpts_heatmap"] = compute_keypoint_map(points, image.shape, self.device) # size=(H,W)
            data["raw"]["valid_mask"] = torch.ones_like(image, device=self.device, dtype=torch.int32) # size=(H,W)


        # Warped pair for SuperPoint (Only for SuperPoint, not MagicPoint)
        if self.config["warped_pair"]:
            assert self.config["has_labels"], "Only for SuperPoint, not MagicPoint."
            
            random_frame_idx = int(self.random_frame(index))

            warped_image = self.samples["image_paths"][random_frame_idx]
            warped_image = self.read_image(warped_image)
            
            warped_name = self.samples["names"][random_frame_idx]

            warped_transformation = np.load(self.samples["camera_transform_paths"][random_frame_idx])
            warped_transformation = self.axis_transform(warped_transformation)
            warped_rotation, warped_translation = self.get_rotation_translation(warped_transformation)

            data["warp"] = {"image":warped_image,
                            "warped_rotation":warped_rotation,
                            "warped_translation":warped_translation}
            
            data["warped_name"] = warped_name

            warped_points = warp_points_NeRF(data["raw"]["kpts"],
                                             data["raw"]["input_depth"].unsqueeze(0),
                                             data["camera_intrinsic_matrix"].unsqueeze(0),
                                             data["raw"]["input_rotation"].unsqueeze(0),
                                             data["raw"]["input_translation"].unsqueeze(0),
                                             data["warp"]["warped_rotation"].unsqueeze(0),
                                             data["warp"]["warped_translation"].unsqueeze(0),
                                             self.device)
            
            warped_points = filter_points(warped_points, warped_image.shape, self.device)
            
            data["warp"]["kpts"] = warped_points
            data["warp"]["kpts_heatmap"] = compute_keypoint_map(warped_points, image.shape, self.device) # size=(H,W)
            data["warp"]["valid_mask"] = torch.ones_like(image,device=self.device,dtype=torch.int32) # size=(H,W)
            
            if self.action == "training" and self.config["augmentation"]["photometric"]["enable"]:
                data["warp"]["image"] = self.photometric_aug(data["warp"]["image"])
                data["warp"]["image"] = torch.as_tensor(data["warp"]["image"], dtype=torch.float32,device=self.device)
            
            data["warp"]["image"] /= 255. # Normalize image to [0,1]

        # Photometric augmentation
        if self.config["has_labels"] and self.action == "training":
            
            if self.config["augmentation"]["photometric"]["enable"]:
                data["raw"]["image"] = self.photometric_aug(data["raw"]["image"])
                data["raw"]["image"] = torch.as_tensor(data["raw"]["image"], dtype=torch.float32, device=self.device)

            if self.config["downsample"]:
                data = self.downsample_data(data)
            
        data["raw"]["image"] /= 255. # Normalize image to [0,1]

        return data
    

    def batch_collator(self, batch):

        
        images = torch.stack([item['raw']['image'].unsqueeze(0) for item in batch]) # size=(batch_size,1,H,W)
        
        input_names = [item['name'] for item in batch]

        output = {"raw": {"image":images},
                  "name":input_names}

        input_depths = torch.stack([item['raw']['input_depth'] for item in batch]) # size=(batch_size,H,W)
    
        input_rotations = torch.stack([item['raw']['input_rotation'] for item in batch]) # size=(batch_size,3,3)
    
        input_translations = torch.stack([item['raw']['input_translation'] for item in batch]) # size=(batch_size,3,1)
    
        intrinsic_matrix = torch.stack([item['camera_intrinsic_matrix'] for item in batch]) # size=(batch_size,3,3)

        output["raw"]["input_depth"] = input_depths # size=(batch_size,H,W)
        output["raw"]["input_rotation"] = input_rotations # size=(batch_size,3,3)
        output["raw"]["input_translation"] = input_translations # size=(batch_size,3,1)           
        output["camera_intrinsic_matrix"] = intrinsic_matrix # size=(batch_size,3,3)

        if self.config["has_labels"]:

            points = [item['raw']['kpts'] for item in batch]

            kp_heatmap = torch.stack([item['raw']['kpts_heatmap'] for item in batch])

            valid_mask = torch.stack([item['raw']['valid_mask'] for item in batch])

            output["raw"]["kpts"] = points # size=(N,2)
            output["raw"]["kpts_heatmap"] = kp_heatmap # size=(batch_size,H,W)
            output["raw"]["valid_mask"] = valid_mask # size=(batch_size,H,W)
        

        if self.config["warped_pair"]:

            warped_images = torch.stack([item['warp']['image'].unsqueeze(0) for item in batch]) # size=(batch_size,1,H,W)
            
            warped_names = [item["warped_name"] for item in batch] 
            
            warped_rotations = torch.stack([item['warp']['warped_rotation'] for item in batch]) # size=(batch_size,3,3)
            
            warped_translations = torch.stack([item['warp']['warped_translation'] for item in batch]) # size=(batch_size,3,1)
            
            warped_points = [item['warp']['kpts'] for item in batch] # size=(N,2)
            
            warped_kp_heatmap = torch.stack([item['warp']['kpts_heatmap'] for item in batch]) # size=(batch_size,H,W)
            
            warped_valid_mask = torch.stack([item['warp']['valid_mask'] for item in batch]) # size=(batch_size,H,W)
        
            
            output["warp"] = {"image":warped_images, # size=(batch_size,1,H,W) 
                              "warped_rotation":warped_rotations, # size=(batch_size,3,3)
                              "warped_translation":warped_translations, # size=(batch_size,3,1)
                              "kpts":warped_points, # size=(N,2)
                              "kpts_heatmap":warped_kp_heatmap, # size=(batch_size,H,W)
                              "valid_mask":warped_valid_mask}
            
            output["warped_name"] = warped_names
        
        return output