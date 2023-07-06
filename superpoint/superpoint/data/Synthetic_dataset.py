import torch
import numpy as np
import cv2
from tqdm import tqdm
import yaml
from pathlib import Path
from torch.utils.data import Dataset
from superpoint.data.data_utils import Synthetic_data
from superpoint.data.data_utils.config_update import dict_update, parse_primitives
from superpoint.data.data_utils.kp_utils import filter_points, compute_keypoint_map
from superpoint.data.data_utils.photometric_augmentation import Photometric_aug
from superpoint.data.data_utils.homographic_augmentation import Homographic_aug
from superpoint.settings import DATA_PATH


class SyntheticShapes(Dataset):

    default_config = {
        'primitives': 'all',
        'truncate': {},
        'suffix': None,
        'add_augmentation_to_test_set': False,
        'generation': {
            'split_sizes': {'training': 10000, 'validation': 200, 'test': 500},
            'image_size': [960, 1280],
            'random_seed': 0,
            'params': {
                'generate_background': {
                    'min_kernel_size': 150, 'max_kernel_size': 500,
                    'min_rad_ratio': 0.02, 'max_rad_ratio': 0.031},
                'draw_stripes': {'transform_params': (0.1, 0.1)},
                'draw_multiple_polygons': {'kernel_boundaries': (50, 100)}
            },
        },
        'preprocessing': {
            'resize': [240, 320],
            'blur_size': 11,
        },
        'augmentation': {
            'photometric': {
                'enable': False,
                'primitives': 'all',
                'params': {},
                'random_order': True,
            },
            'homographic': {
                'enable': False,
                'params': {},
                'valid_border_margin': 0,
            },
        }
    }
    drawing_primitives = [
        'draw_lines',
        'draw_polygon',
        'draw_multiple_polygons',
        'draw_ellipses',
        'draw_star',
        'draw_checkerboard',
        'draw_stripes',
        'draw_cube',
        'gaussian_noise'
    ]

    def __init__(self,config, task = "train" ,device="cpu") -> None:
        super(SyntheticShapes,self).__init__()

        self.config = self.default_config
        self.config = dict_update(self.config, dict(config))
        self.device = device
        self.action = ["training"] if task == "train" else ["validation"] if task == "val" else ["test"]
        self.samples = self._init_dataset()
        self.photometric_aug = Photometric_aug(self.config["augmentation"]["photometric"])
        self.homographic_aug = Homographic_aug(self.config["augmentation"]["homographic"], device=self.device)


    def dump_primitive_data(self, primitive):

        output_dir = Path(DATA_PATH,self.config['data_dir'], primitive)
        
        Synthetic_data.set_random_state(
            np.random.RandomState(self.config["generation"]["random_seed"])
        )

        for split, size in self.config["generation"]["split_sizes"].items():
            
            im_dir, pts_dir = [Path(output_dir, i, split) for i in ["images", "points"]]
            im_dir.mkdir(parents=True, exist_ok=True)
            pts_dir.mkdir(parents=True, exist_ok=True)

            for i in tqdm(range(size), desc=split, leave=False):

                image = Synthetic_data.generate_background(
                    self.config["generation"]["image_size"],
                    **self.config["generation"]["params"]["generate_background"],
                )

                points = np.array(
                    getattr(Synthetic_data, primitive)(
                        image, **self.config["generation"]["params"].get(primitive, {})
                    )
                )

                points = np.flip(points, 1)  # reverse convention with opencv

                b = self.config["preprocessing"]["blur_size"]
                image = cv2.GaussianBlur(image, (b, b), 0)
                points = (
                    points
                    * np.array(self.config["preprocessing"]["resize"], np.float32)
                    / np.array(self.config["generation"]["image_size"], np.float32)
                )
                image = cv2.resize(
                    image,
                    tuple(self.config["preprocessing"]["resize"][::-1]),
                    interpolation=cv2.INTER_LINEAR,
                )

                cv2.imwrite(str(Path(im_dir, "{}.png".format(i))), image)
                np.save(Path(pts_dir, "{}.npy".format(i)), points)


    def _init_dataset(self):
        primitives = parse_primitives(self.config["primitives"], self.drawing_primitives)

        basepath = Path(DATA_PATH,self.config['data_dir'])
        basepath.mkdir(parents=True, exist_ok=True)

        data = []
        for primitive in primitives:
            primitive_dir = Path(basepath, primitive)
            if not primitive_dir.exists():
                self.dump_primitive_data(primitive)

            truncate = self.config['truncate'].get(primitive, 1)
            
            for s in self.action:
                e = [str(p) for p in Path(primitive_dir, 'images', s).iterdir()]
                f = [p.replace('images', 'points') for p in e]
                f = [p.replace('.png', '.npy') for p in f]
                data.extend([{"image": _im, "point": _pt } 
                             for _im, _pt in 
                             zip(e[: int(truncate * len(e))],f[: int(truncate * len(f))])    
                             ])

        return data


    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, index):
        sample = self.samples[index]

        image = cv2.imread(sample["image"], cv2.IMREAD_GRAYSCALE)

        image = torch.as_tensor(image, dtype=torch.float32,device=self.device) #size=(H,W)
        
        H,W = image.size()
        
        points = np.load(sample["point"]) # load points
        points = torch.as_tensor(points, dtype=torch.float32,device=self.device) # coordinates=(y,x), convert to tensor
        points = filter_points(points, torch.tensor([H,W],device=self.device), return_mask=False) # filter points at border.
        kp_map = compute_keypoint_map(points, image.size(), device=self.device) # create keypoint map/mask where keypoints
                                                                                # coordinates are 1 and the rest are 0
        valid_mask = torch.ones_like(image) # size=(H,W)
        homography = torch.eye(3, device=self.device) # size=(H,W)
       
        data = {'raw':{'image':image, # size =(H,W)
                       'kpts':points, # size=(N,2)
                       'kpts_map':kp_map, # size =(H,W)
                       'mask':valid_mask, # size =(H,W)
                        },
                'homography': homography} # size = (3,3)
        
        if (self.config["augmentation"]["photometric"]["enable_train"] and self.action[0] == "training" or
            self.config["augmentation"]["photometric"]["enable_val"] and self.action[0] == "validation"):
            
            image = self.photometric_aug(data['raw']['image']) # size=(H,W)
            data['raw']['image'] = torch.tensor(image, dtype=torch.float32,device=self.device) # size=(H,W) 
                       
        
        if (self.config["augmentation"]["homographic"]["enable_train"] and self.action[0] == "training" or
            self.config["augmentation"]["homographic"]["enable_val"] and self.action[0] == "validation"):

            image = data['raw']['image'].view(1,1,H,W)
            keypoints = data['raw']['kpts']
            warped = self.homographic_aug(image, keypoints)
            
            data["raw"] = warped["warp"]
            data["homography"] = warped["homography"]        

        data['raw']['image'] /= 255.0

        
        if len(data['raw']['image'].size()) != 3:
            data['raw']['image'] = data['raw']['image'].view(1,H,W)
                
        return data
    
    def batch_collator(self, batch):
        assert(len(batch)>0 and isinstance(batch[0], dict))
        
        images = torch.stack([item['raw']['image'] for item in batch])
        points = [item['raw']['kpts'] for item in batch]
        kp_map = torch.stack([item['raw']['kpts_map'] for item in batch])
        mask = torch.stack([item['raw']['mask'] for item in batch])
        homography = torch.stack([item['homography'] for item in batch])

        batch = {'raw':{'image':images, # size=(batch_size,1,H,W)
                'kpts': points, # size=(N,2)
                'kpts_map': kp_map, # size=(batch_size,H,W)
                'mask': mask, # size=(batch_size,H,W)
                },
                'homography': homography} #size=(batch_size,1,3,3)
        
        return batch



    
