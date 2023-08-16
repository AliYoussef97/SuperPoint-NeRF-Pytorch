from pathlib import Path
import os
import torch
import kornia.geometry.transform as K
import kornia
import cv2
import numpy as np
from tqdm import tqdm
from superpoint.data.data_utils.homographic_augmentation import Homographic_aug
from superpoint.models.model_utils.sp_utils import box_nms
from superpoint.settings import EXPER_PATH
from superpoint.utils.train_utils import move_to_device

class ExportDetections():
    def __init__(self, config, model, dataloader, split, enable_HA, device):
        
        self.config = config
        self.model = model.eval()
        self.dataloader = dataloader
        self.split = split
        self.enable_HA = enable_HA
        if self.enable_HA:
            print(f'\033[92m✅ Homography Adaptation enabled \033[0m')
        self.device = device
        self.output_dir = self._init_output_dir()
        self.one_homography = Homographic_aug(config['homography_adaptation'],self.device)
        self.homography_adaptation()

    def _init_output_dir(self):
        """
        Where to save the outputs.
        """
        output_dir = Path(EXPER_PATH, 'outputs', self.config['data']['experiment_name'], self.split)
        if not output_dir.exists():
            os.makedirs(output_dir)
        return output_dir
    

    @torch.no_grad()
    def step(self,image,probs,counts):
        
        image_shape = image.shape[2:]
        
        H = self.one_homography.sample_homography(shape=image_shape,
                                                  **self.config['homography_adaptation']['params']) # 1,3,3
        H_inv = torch.inverse(H) # 1,3,3

        warped_image = K.warp_perspective(image, H, dsize=(image_shape), align_corners=True) # 1,1,H,W
       
        mask = K.warp_perspective(torch.ones_like(warped_image,device=self.device), H, dsize=(image_shape), mode='nearest', align_corners=True) # 1,1,H,W

        count = K.warp_perspective(torch.ones_like(warped_image,device=self.device), H_inv, dsize=(image_shape), mode='nearest', align_corners=True) # 1,1,H,W

        if self.config['homography_adaptation']['valid_border_margin']:
            erosion = self.config['homography_adaptation']['valid_border_margin']
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion*2,)*2)
            kernel = torch.as_tensor(kernel,device=self.device, dtype=torch.float32)
            
            mask = kornia.morphology.erosion(mask,kernel).to(torch.int32) # 1,1,H,W
            mask = mask.squeeze(1) # 1,H,W

            count = kornia.morphology.erosion(count,kernel).to(torch.int32) # 1,1,H,W
            count = count.squeeze(1) # 1,H,W

        
        prob = self.model(warped_image)['detector_output']['prob_heatmap'] # 1,H,W
        prob *= mask

        prob_proj = K.warp_perspective(prob.unsqueeze(0), H_inv, dsize=(image_shape), mode='bilinear', align_corners=True) # 1,1,H,W
        prob_proj = prob_proj.squeeze(1) # 1,H,W
        prob_proj *= count # 1,H,W
        
        probs = torch.concat([probs, prob_proj.unsqueeze(1)], dim=1)
        counts = torch.concat([counts, count.unsqueeze(1)], dim=1)
        
        return probs, counts   

    
    @torch.no_grad()
    def homography_adaptation(self):
        for data in tqdm(self.dataloader, desc=f"Exporting detections",colour="green"):

            data = move_to_device(data,self.device)

            name = data["name"][0]
            save_path = Path(self.output_dir, '{}.npy'.format(name))
            if save_path.exists():
                continue
           
            probs = self.model(data["raw"]["image"])["detector_output"]["prob_heatmap"] # 1,H,W
            

            if self.enable_HA:

                counts = torch.ones_like(probs,device=self.device) # 1,H,W
                
                probs = probs.unsqueeze(1) # 1,1,H,W
                counts = counts.unsqueeze(1) # 1,1,H,W
                
                for _ in range(self.config["homography_adaptation"]["num"]-1):
                    probs, counts = self.step(data["raw"]["image"], probs, counts) # 1,num,H,W, 1,num,H,W

                counts = torch.sum(counts, dim=1) # 1,H,W
                max_prob, _ = torch.max(probs, dim=1) # 1,H,W
                mean_prob = torch.sum(probs, dim=1) / counts # 1,H,W
            
                if self.config["homography_adaptation"]["aggregation"] == "max":
                    probs = max_prob # 1,H,W
                
                if self.config["homography_adaptation"]["aggregation"] == "sum":
                    probs = mean_prob # 1,H,W
            
            probs = [box_nms(prob=pb,
                             size=self.config["model"]["detector_head"]["nms"],
                             min_prob=self.config["model"]["detector_head"]["det_thresh"],
                             keep_top_k=self.config["model"]["detector_head"]["top_k"]) for pb in probs]
            
            probs = torch.stack(probs) # 1,H,W
            
            pred = torch.ge(probs,self.config["model"]["detector_head"]["det_thresh"]).to(torch.int32) # 1,H,W

            pred = torch.nonzero(pred.squeeze(0), as_tuple=False) # N,2

            pred = pred.cpu().numpy()

            np.save(save_path, pred)




class Export_Hpatches_Repeatability():
    def __init__(self, config, model, dataloader, device):
        
        self.config = config
        self.model = model.eval()
        self.dataloader = dataloader
        self.device = device
        self.output_dir = self._init_output_dir()
        self.export_repeatability()

    
    def _init_output_dir(self):
        """
        Where to save the outputs.
        """
        output_dir = Path(EXPER_PATH, 'repeatability', self.config['data']['experiment_name'])
        if not output_dir.exists():
            os.makedirs(output_dir)
        return output_dir

    @torch.no_grad()
    def export_repeatability(self):
        
        for i, data in enumerate(tqdm(self.dataloader, desc=f"Exporting repeatability detections", colour="green")):

            data = move_to_device(data,self.device)

            prob1 = self.model(data["image"])["detector_output"]["prob_heatmap_nms"] # 1,H,W
            prob2 = self.model(data["warped_image"])["detector_output"]["prob_heatmap_nms"] # 1,H,W

            output = {"image": data["image"].squeeze().cpu().numpy(),
                      "warped_image": data["warped_image"].squeeze().cpu().numpy(),
                      "prob": prob1.squeeze().cpu().numpy(),
                      "warped_prob": prob2.squeeze().cpu().numpy(),
                      "homography": data["homography"].squeeze().cpu().numpy()}

            filename = data['name'][0] if 'name' in data else str(i)
            save_path = Path(self.output_dir, '{}.npz'.format(filename))
            np.savez_compressed(save_path, **output)




class Export_Hpatches_Descriptors():
    def __init__(self, config, model, dataloader, device):
        
        self.config = config
        self.model = model.eval()
        self.dataloader = dataloader
        self.device = device
        self.output_dir = self._init_output_dir()
        self.export_descriptors()

    
    def _init_output_dir(self):
        """
        Where to save the outputs.
        """
        output_dir = Path(EXPER_PATH, 'descriptors', self.config['data']['experiment_name'])
        if not output_dir.exists():
            os.makedirs(output_dir)
        return output_dir

    @torch.no_grad()
    def export_descriptors(self):
        
        for i, data in enumerate(tqdm(self.dataloader, desc=f"Exporting HPatches descriptors", colour="green")):

            data = move_to_device(data,self.device)

            output_1 = self.model(data["image"])
            prob1 = output_1["detector_output"]["prob_heatmap_nms"]
            desc1 = output_1["descriptor_output"]["desc"] 

            output_2 = self.model(data["warped_image"])
            prob2 = output_2["detector_output"]["prob_heatmap_nms"] 
            desc2 = output_2["descriptor_output"]["desc"] 

            output = {"image": data["image"].squeeze().cpu().numpy(),
                      "warped_image": data["warped_image"].squeeze().cpu().numpy(),
                      "prob": prob1.squeeze().cpu().numpy(),
                      "warped_prob": prob2.squeeze().cpu().numpy(),
                      "desc": desc1.squeeze().cpu().numpy().transpose(1,2,0),
                      "warped_desc": desc2.squeeze().cpu().numpy().transpose(1,2,0),
                      "homography": data["homography"].squeeze().cpu().numpy()}

            filename = data['name'][0] if 'name' in data else str(i)
            save_path = Path(self.output_dir, "{}.npz".format(filename))
            np.savez_compressed(save_path, **output)