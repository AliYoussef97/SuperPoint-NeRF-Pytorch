from torch.utils.data import Dataset
import torch
import numpy as np
from superpoint.settings import DATA_PATH
from pathlib import Path


class NeRF_Dataset(Dataset):
    def __init__(self, config, task = "train" ,device="cpu") -> None:
        super(NeRF_Dataset, self).__init__()
        self.config = config
        self.device = device
        self.action = ["training"] if task == "train" else ["validation"] if task == "validation" else ["test"]
        self.depth = torch.tensor(np.load(Path(DATA_PATH, self.config['data_dir'], 'depth_meters.npy')))
        self.samples = self._init_dataset()
    
    def _init_dataset(self):

        basepath = Path(DATA_PATH,self.config['data_dir'])
        basepath.mkdir(parents=True, exist_ok=True)
        
        data = []

        for s in self.action:
            e = [str(p) for p in Path(basepath, 'images', s).iterdir()]
            f = [p.replace('images', 'points') for p in e]
            f = [p.replace('.png', '.npy') for p in f]
            data.extend([{"image": _im, "point": _pt } 
                            for _im, _pt in 
                            zip(e,f)    
                            ])

        return data

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        1


