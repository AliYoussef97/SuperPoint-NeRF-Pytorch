# Parts of code inspiried from SuperPoint[https://github.com/rpautrat/SuperPoint]

import cv2
import numpy as np


class Photometric_aug():
    def __init__(self,config)-> np.ndarray:
        
        self.primitives = config['primitives']
        self.params = config['params']

    
    def random_brightness(self, image, max_abs_change=75):
        delta = np.random.uniform(low=-max_abs_change,high=max_abs_change, size=1)[0]
        image = image + delta
        image = np.clip(image, 0, 255.0)
        return image.astype(np.float32)
            

    def random_contrast(self, image, strength_range=(0.3, 1.8)):
        contrast_factor = np.random.uniform(low=strength_range[0],
                                            high=strength_range[1],
                                            size=1)[0]
        mean = image.mean()
        image = (image-mean)*contrast_factor+mean
        image = np.clip(image, 0, 255.)
        return image.astype(np.float32)
            

    def additive_gaussian_noise(self, image, stddev_range=(0, 15)):
        stddev = np.random.uniform(stddev_range[0], stddev_range[1])
        noise = np.random.normal(scale=stddev,size=image.shape)
        noisy_image = np.clip(image+noise, 0, 255)
        return noisy_image


    def additive_speckle_noise(self, image, prob_range=(0, 0.0035)):

        prob = np.random.uniform(prob_range[0], prob_range[1])
        sample = np.random.uniform(size=image.shape)
        noisy_image = np.where(sample<=prob, np.zeros_like(image), image)
        noisy_image = np.where(sample>=(1. - prob), 255.*np.ones_like(image), noisy_image)
        noisy_image = np.clip(noisy_image.round(),0,255)
        return noisy_image
    

    def motion_blur(self, image, max_kernel_size=7):

        def _py_motion_blur(img):
            # Either vertial, hozirontal or diagonal blur
            mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
            ksize = np.random.randint(0, (max_kernel_size+1)/2)*2 + 1  # make sure is odd
            center = int((ksize-1)/2)
            kernel = np.zeros((ksize, ksize))
            if mode == 'h':
                kernel[center, :] = 1.
            elif mode == 'v':
                kernel[:, center] = 1.
            elif mode == 'diag_down':
                kernel = np.eye(ksize)
            elif mode == 'diag_up':
                kernel = np.flip(np.eye(ksize), 0)
            var = ksize * ksize / 16.
            grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
            gaussian = np.exp(-(np.square(grid-center)+np.square(grid.T-center))/(2.*var))
            kernel *= gaussian
            kernel /= np.sum(kernel)
            img = cv2.filter2D(img, -1, kernel)
            return img

        blurred = _py_motion_blur(image)
        res = np.reshape(blurred, image.shape)
        return res


    def additive_shade(self, image, kernel_size_range=(50, 100), transparency_range=[-0.5, 0.8], nb_ellipses=20):
  
        def _py_additive_shade(img):
            min_dim = min(img.shape[:2]) / 4
            mask = np.zeros(img.shape[:2], np.uint8)
            for i in range(nb_ellipses):
                ax = int(max(np.random.rand() * min_dim, min_dim / 5))
                ay = int(max(np.random.rand() * min_dim, min_dim / 5))
                max_rad = max(ax, ay)
                x = np.random.randint(max_rad, img.shape[1] - max_rad)  # center
                y = np.random.randint(max_rad, img.shape[0] - max_rad)
                angle = np.random.rand() * 90
                cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

            transparency = np.random.uniform(*transparency_range)
            kernel_size = np.random.randint(*kernel_size_range)
            if (kernel_size % 2) == 0:  # kernel_size has to be odd
                kernel_size += 1
            mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
            shaded = img * (1 - transparency * mask/255.)
            
            return np.clip(shaded, 0, 255)

        shaded = _py_additive_shade(image)
        res = np.reshape(shaded, image.shape)
   
        return res


    def __call__(self, image):
        
        image = image.cpu().numpy().astype(np.uint8)

        indices = np.arange(len(self.primitives))
        np.random.shuffle(indices)

        for i in indices:
            primitive = self.primitives[i]
            image = getattr(self, primitive)(image, **self.params[primitive])

        image = image.astype(np.float32)
        
        return image
