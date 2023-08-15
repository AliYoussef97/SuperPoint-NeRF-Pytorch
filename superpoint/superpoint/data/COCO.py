import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.functional as TF
import kornia.geometry.transform as K
from superpoint.data.data_utils.kp_utils import compute_keypoint_map
from superpoint.data.data_utils.photometric_augmentation import Photometric_aug
from superpoint.data.data_utils.homographic_augmentation import Homographic_aug
from superpoint.settings import DATA_PATH, EXPER_PATH


class COCO(Dataset):
    def __init__(self, data_config, task = "training", device="cpu") -> None:
        super(COCO,self).__init__()

        self.config = data_config
        self.device = device
        self.action = "training" if task == "training" else "validation" if task == "validation" else "test"
        self.samples = self._init_dataset()

        if self.config["augmentation"]["photometric"]["enable"]:
            self.photometric_aug = Photometric_aug(self.config["augmentation"]["photometric"])
        
        if self.config["augmentation"]["homographic"]["enable"]:
            self.homographic_aug = Homographic_aug(self.config["augmentation"]["homographic"], device=self.device)
        
        if self.config["warped_pair"]:
            self.homographic_aug = Homographic_aug(self.config["augmentation"]["pair_homography"], device=self.device)


    def _init_dataset(self):
        """
        List of images' path and names to be processed.
        """
        data_dir = Path(DATA_PATH, self.config["name"], "images", self.action)
        image_paths = list(data_dir.iterdir())
        if self.config["truncate"]:
            image_paths = image_paths[:int(self.config["truncate"]*len(image_paths))]
        names = [p.stem for p in image_paths]
        image_paths = [str(p) for p in image_paths]
        files = {"image_paths":image_paths, "names":names}

        if self.config["has_labels"]:

            label_dir = Path(EXPER_PATH, self.config["has_labels"], self.action)
            label_paths = []
            for n in files["names"]:
                p = Path(label_dir,'{}.npy'.format(n))
                label_paths.append(str(p))
            files["label_paths"] = label_paths
        
        return files


    def __len__(self):
        return len(self.samples["image_paths"])
    

    def read_image(self, image):
        image = torchvision.io.read_file(image)
        image = torchvision.io.decode_image(image,torchvision.io.ImageReadMode.GRAY)
        return image.squeeze(0).to(torch.float32).to(self.device)
    
    
    def ratio_preserving_resize(self, image):
        """
        Resize image while preserving the aspect ratio.
        """
        target_size = torch.as_tensor(self.config["preprocessing"]["resize"], dtype=torch.int32)
        scales = torch.divide(target_size, torch.as_tensor(image.shape, dtype=torch.float32))
        new_size = (torch.as_tensor(image.shape[:2], dtype=torch.float32) * torch.max(scales)).to(torch.int32)
        image = K.resize(image,size=[new_size[0], new_size[1]], interpolation='bilinear' ,align_corners=False)
        image = TF.center_crop(image,output_size=[target_size[0].item(), target_size[1].item()])      
        return image
    

    def __getitem__(self, index):
        """
        Note:
            The input image is resized to the size specified in the configuration file.
            If the model is MagicPoint, the input image has Photometric and Homographic augmentations.
            And there is no warped pair.

            If the model is SuperPoint, the input image has only Photometric augmentation. While,
            the warped image has Photometric and Homographic augmentations.
        """

        image = self.samples["image_paths"][index]
        image = self.read_image(image)
        image = self.ratio_preserving_resize(image)
        H,W = image.size()
        data = {'raw':{'image': image},
                'name': self.samples["names"][index]}

        # Add labels if exists.
        if self.config["has_labels"]: # Only for training/validaiton/test not exporting pseudo labels.
            points = self.samples["label_paths"][index]
            points = np.load(points)
            points = torch.as_tensor(points, dtype=torch.float32, device=self.device)
            data["raw"]["kpts"] = points
            data["raw"]["kpts_heatmap"] = compute_keypoint_map(points, image.shape, self.device) # size=(H,W)
            data["raw"]["valid_mask"] = torch.ones_like(image,device=self.device,dtype=torch.int32) # size=(H,W)
            data["homography"] = torch.eye(3, device=self.device) # size=(3,3)
        

        # Warped pair for SuperPoint (Only for SuperPoint, not MagicPoint)
        if self.config["warped_pair"]:
            assert self.config["has_labels"], "Only for SuperPoint, not MagicPoint."

            warped = self.homographic_aug(data["raw"]["image"].view(1,1,H,W),data["raw"]["kpts"])
            data["warp"] = warped["warp"]
            data["homography"] = warped["homography"]
            
            if self.action == "training" and self.config["augmentation"]["photometric"]["enable"]:
                data["warp"]["image"] = self.photometric_aug(data["warp"]["image"])
                data["warp"]["image"] = torch.as_tensor(data["warp"]["image"], dtype=torch.float32,device=self.device)
            
            data["warp"]["image"] /= 255. # Normalize image to [0,1]
        
        
        # Data augmentation
        if self.config["has_labels"] and self.action == "training":
            
            if self.config["augmentation"]["photometric"]["enable"]:
                data["raw"]["image"] = self.photometric_aug(data["raw"]["image"])
                data["raw"]["image"] = torch.as_tensor(data["raw"]["image"], dtype=torch.float32, device=self.device)
            
            if self.config["augmentation"]["homographic"]["enable"]:
                assert not self.config["warped_pair"], "Only for MagicPoint, not SuperPoint."
                data_o = self.homographic_aug(data["raw"]["image"].view(1,1,H,W), data["raw"]["kpts"])
                data["raw"] = data_o["warp"]
                data["homography"] = data_o["homography"]
        
        data["raw"]["image"] /= 255. # Normalize image to [0,1]

        return data
    
    
    def batch_collator(self, batch):
        assert(len(batch)>0 and isinstance(batch[0], dict))
        
        images = torch.stack([item['raw']['image'].unsqueeze(0) for item in batch]) # size=(batch_size,1,H,W)
        names = [item['name'] for item in batch]
        
        output = {'raw': {'image': images},
                  'name': names}
        
        if self.config["has_labels"]:
            
            points = [item['raw']['kpts'] for item in batch]
            kp_heatmap = torch.stack([item['raw']['kpts_heatmap'] for item in batch])
            valid_mask = torch.stack([item['raw']['valid_mask'] for item in batch])
            
            output["raw"]["kpts"] = points # size=(N,2)
            output["raw"]["kpts_heatmap"] = kp_heatmap # size=(batch_size,H,W)
            output["raw"]["valid_mask"] = valid_mask # size=(batch_size,H,W)


        if self.config["warped_pair"]:
            
            warped_images = torch.stack([item['warp']['image'].unsqueeze(0) for item in batch])
            warped_points = [item['warp']['kpts'] for item in batch]
            warped_kp_heatmap = torch.stack([item['warp']['kpts_heatmap'] for item in batch])
            warped_valid_mask = torch.stack([item['warp']['valid_mask'] for item in batch])
        
            output['warp'] = {'image': warped_images, # size=(batch_size,1,H,W)
                              'kpts': warped_points, # size=(N,2)
                              'kpts_heatmap': warped_kp_heatmap, # size=(batch_size,H,W)
                              'valid_mask': warped_valid_mask, # size=(batch_size,H,W)
                             }
        
        if self.config["has_labels"] or self.config["warped_pair"]: 
            """
            Only when not exporting pseudo labels.
            For training, if warped pair is enabled, the homography is of the warped pair.
            If warped pair is not enabled, the homography is of the input image.
            For validation and test, the homography is an identity matrix.
            """
            homography = torch.stack([item['homography'] for item in batch])
            output["homography"] = homography # size=(batch_size,3,3)

        
        return output