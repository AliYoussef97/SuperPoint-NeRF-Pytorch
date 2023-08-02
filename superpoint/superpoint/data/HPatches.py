import torch
import torchvision.transforms.functional as TF
import kornia.geometry.transform as K
import cv2
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from superpoint.settings import DATA_PATH
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class HPatches(Dataset):
    def __init__(self, data_config, device="cpu") -> None:
        super(HPatches,self).__init__()

        self.config = data_config
        self.device = device
        self.samples = self._init_dataset()
    

    def _init_dataset(self):

        data_dir = Path(f"{DATA_PATH}\\{self.config['name']}")
        folder_dirs = [x for x in data_dir.iterdir() if x.is_dir()]

        image_paths = []
        warped_image_paths = []
        homographies = []
        names = []

        for folder_dir in folder_dirs:
            if self.config["alteration"] == 'i' != folder_dir.stem[0] != 'i':
                continue
            if self.config["alteration"] == 'v' != folder_dir.stem[0] != 'v':
                continue

            num_images = 1 if self.config['name'] == 'COCO' else 5
            file_ext = '.ppm' if self.config['name'] == 'HPatches' else '.jpg'   

            for i in range(2, 2 + num_images):
                image_paths.append(str(Path(folder_dir, "1" + file_ext)))
                warped_image_paths.append(str(Path(folder_dir, str(i) + file_ext)))
                homographies.append(np.loadtxt(str(Path(folder_dir, "H_1_" + str(i)))))
                names.append(f"{folder_dir.stem}_{1}_{i}")
        
        files = {'image_paths': image_paths,
                 'warped_image_paths': warped_image_paths,
                 'homography': homographies,
                 'names': names} 
        
        return files


    def __len__(self):
        return len(self.samples['image_paths'])
    

    def read_image(self, image):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        return torch.as_tensor(image, dtype=torch.float32, device=self.device)
    
    
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
    
    def adapt_homography_to_resize(self, homographies):
        source_size = homographies["image_shape"]
        source_warped_size = homographies["warped_image_shape"]
        target_size = torch.as_tensor(self.config["preprocessing"]["resize"], dtype=torch.float32, device=self.device)

        s = torch.max(torch.divide(target_size, source_size))
        up_scale = torch.diag(torch.stack([1. / s, 1. / s, torch.tensor(1.)]))
        warped_s = torch.max(torch.divide(target_size, source_warped_size))
        down_scale = torch.diag(torch.stack([warped_s, warped_s, torch.tensor(1.)]))
        
        pad_y = ((source_size[0] * s - target_size[0]) / torch.tensor(2.)).to(torch.int32)
        pad_x = ((source_size[1] * s - target_size[1]) / torch.tensor(2.)).to(torch.int32)
        translation = torch.diag(torch.stack([torch.tensor(1.), torch.tensor(1.), torch.tensor(1.)]))    
        translation[0, -1] = pad_x
        translation[1, -1] = pad_y

        pad_y = ((source_warped_size[0] * warped_s - target_size[0]) / torch.tensor(2.)).to(torch.int32)
        pad_x = ((source_warped_size[1] * warped_s - target_size[1]) / torch.tensor(2.)).to(torch.int32)
        warped_translation = torch.diag(torch.stack([torch.tensor(1.), torch.tensor(1.), torch.tensor(1.)]))
        warped_translation[0, -1] = -pad_x
        warped_translation[1, -1] = -pad_y

        H = warped_translation @ down_scale @ homographies["homography"] @ up_scale @ translation

        return H
    

    def __getitem__(self, index):

        image = self.read_image(self.samples['image_paths'][index])
        warped_image = self.read_image(self.samples['warped_image_paths'][index])
        homography = torch.as_tensor(self.samples['homography'][index], dtype=torch.float32, device=self.device)
        name = self.samples['names'][index]

        if self.config["preprocessing"]["resize"]:
           
            image_shape = torch._shape_as_tensor(image)
            warped_image_shape = torch._shape_as_tensor(warped_image)

            homographies = {"homography": homography,
                            "image_shape": image_shape,
                            "warped_image_shape": warped_image_shape}
           
            homography = self.adapt_homography_to_resize(homographies)

        
        image = self.ratio_preserving_resize(image)
        warped_image = self.ratio_preserving_resize(warped_image)

        image /= 255.
        warped_image /= 255.

        data = {"image": image,
                "warped_image": warped_image,
                "homography": homography,
                "name": name}
        
        return data
    

    def batch_collator(self,batch):
        
        images = torch.stack([item["image"].unsqueeze(0) for item in batch])
        warped_images = torch.stack([item["warped_image"].unsqueeze(0) for item in batch])
        homographies = torch.stack([item["homography"] for item in batch])
        names = [item["name"] for item in batch]

        output = {"image": images,
                  "warped_image": warped_images,
                  "homography": homographies,
                  "name": names}
        
        return output