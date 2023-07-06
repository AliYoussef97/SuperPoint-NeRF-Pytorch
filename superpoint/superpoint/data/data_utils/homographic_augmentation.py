import cv2
import numpy as np
import torch
from numpy.random import uniform
from scipy.stats import truncnorm
from superpoint.data.data_utils.config_update import dict_update
from superpoint.data.data_utils.kp_utils import filter_points, compute_keypoint_map, warp_points
import kornia.geometry.transform as tf
import kornia


class Homographic_aug():
    def __init__(self, config, device="cpu") -> dict:
        self.config = config["params"]
        self.erosion = config["valid_border_margin"]
        self.device = device


    def sample_homography(self, shape, translation=True, rotation=True, scaling=True, perspective=True, scaling_amplitude=0.2,
                          n_scales=5, n_angles=25, perspective_amplitude_x=0.2,perspective_amplitude_y=0.2,
                          patch_ratio=0.8,max_angle=1.57,allow_artifacts=True,translation_overflow=0.05):
        
        std_trunc = 2

        # Corners of the input patch
        margin = (1 - patch_ratio) / 2
        pts1 = margin + np.array([[0, 0],
                                [0, patch_ratio],
                                [patch_ratio, patch_ratio],
                                [patch_ratio, 0]])
        pts2 = pts1.copy()

        # Random perspective and affine perturbations
        if perspective:
            if not allow_artifacts:
                perspective_amplitude_x = min(perspective_amplitude_x, margin)
                perspective_amplitude_y = min(perspective_amplitude_y, margin)
            else:
                perspective_amplitude_x = perspective_amplitude_x
                perspective_amplitude_y = perspective_amplitude_y
            perspective_displacement = truncnorm(-std_trunc, std_trunc, loc=0., scale=perspective_amplitude_y/2).rvs(1)
            h_displacement_left = truncnorm(-std_trunc, std_trunc, loc=0., scale=perspective_amplitude_x/2).rvs(1)
            h_displacement_right = truncnorm(-std_trunc, std_trunc, loc=0., scale=perspective_amplitude_x/2).rvs(1)
            pts2 += np.array([[h_displacement_left, perspective_displacement],
                            [h_displacement_left, -perspective_displacement],
                            [h_displacement_right, perspective_displacement],
                            [h_displacement_right, -perspective_displacement]]).squeeze()

        # Random scaling
        # sample several scales, check collision with borders, randomly pick a valid one
        if scaling:
            scales = truncnorm(-std_trunc, std_trunc, loc=1, scale=scaling_amplitude/2).rvs(n_scales)
            scales = np.concatenate((np.array([1]), scales), axis=0)

            center = np.mean(pts2, axis=0, keepdims=True)
            scaled = (pts2 - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
            if allow_artifacts:
                valid = np.arange(n_scales)  # all scales are valid except scale=1
            else:
                valid = (scaled >= 0.) * (scaled < 1.)
                valid = valid.prod(axis=1).prod(axis=1)
                valid = np.where(valid)[0]
            idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
            pts2 = scaled[idx,:,:]


        # Random translation
        if translation:
            t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
            if allow_artifacts:
                t_min += translation_overflow
                t_max += translation_overflow
            pts2 += np.array([uniform(-t_min[0], t_max[0],1), uniform(-t_min[1], t_max[1], 1)]).T


        # Random rotation
        # sample several rotations, check collision with borders, randomly pick a valid one
        if rotation:
            angles = np.linspace(-max_angle, max_angle, num=n_angles)
            angles = np.concatenate((np.array([0.]),angles), axis=0)  # in case no rotation is valid
            center = np.mean(pts2, axis=0, keepdims=True)
            rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                        np.cos(angles)], axis=1), [-1, 2, 2])
            rotated = np.matmul( (pts2 - center)[np.newaxis,:,:], rot_mat) + center

            if allow_artifacts:
                valid = np.arange(n_angles)  # all scales are valid except scale=1
            else:
                valid = (rotated >= 0.) * (rotated < 1.)
                valid = valid.prod(axis=1).prod(axis=1)
                valid = np.where(valid)[0]
            idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
            pts2 = rotated[idx,:,:]

        # Rescale to actual size
        shape = np.array(shape[::-1])  # different convention [y, x]
        pts1 *= shape[np.newaxis,:]
        pts2 *= shape[np.newaxis,:]

        # this homography is the same with tf version and this line
        homography = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
        homography = torch.tensor(homography,device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        homography = torch.inverse(homography)
     
        return homography
    

    def compute_valid_mask(self, shape, homography, erosion=2):
        if len(homography.shape)==2:
            homography = homography.unsqueeze(0)
        
        batch_size = homography.shape[0]

        image = torch.ones(tuple([batch_size,1,*shape]),device=self.device, dtype=torch.float32)
        mask = tf.warp_perspective(image, homography, shape, mode="nearest",align_corners=True)

        if erosion>0:

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion*2,)*2)
            kernel = torch.tensor(kernel,device=self.device, dtype=torch.float32)
            origin = ((kernel.shape[0]-1)//2, (kernel.shape[1]-1)//2)
            mask = kornia.morphology.erosion(mask,kernel,origin=origin)

        return mask.squeeze(1)

    


    def __call__(self, image, points):
        
        image_shape = image.shape[2:]
        
        homography = self.sample_homography(shape=image_shape,**self.config) # size= (1,3,3)
        
        warped_image = tf.warp_perspective(image, homography, (image_shape), mode="bilinear" ,align_corners=True) # size = (1,1,H,W)
        warped_image = warped_image.view(1,*image_shape) # size = (1,H,W)
        
        warped_points = warp_points(points, homography, device=self.device) # size = (N,2)
        warped_points = filter_points(warped_points, torch.tensor(image_shape,device=self.device)) # size = (N,2)
        
        warped_points_map = compute_keypoint_map(warped_points, image.shape[2:], device=self.device) # size = (H,W)

        warped_valid_mask = self.compute_valid_mask(image_shape, homography, erosion=self.erosion).squeeze(0) #size = (H,W)
       
        data = {'warp':{'image': warped_image, #(1,H,W)
                        'kpts': warped_points, #(N,2)
                        'kpts_map': warped_points_map, #(H,W)
                        'mask':warped_valid_mask}, #(H,W)
                'homography':homography, #(1,3,3)
                }
        
        return data
