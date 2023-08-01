import tyro
import yaml
import torch
from typing import Literal
from dataclasses import dataclass 
from superpoint.settings import CKPT_PATH
from superpoint.utils.get_model import get_model
from superpoint.utils.data_loaders import get_loader
from superpoint.engine_solvers.train import train_val
from superpoint.engine_solvers.export import ExportDetections


@dataclass
class options:
    """Training options.

    Args:
        validate_training: Validate during training.
        include_mask_loss: Apply mask or no mask during loss (Do not include bordering artifacts by applying mask).
        nerf_loss: Whether to use Descriptor NeRF loss or normal SuperPoint Descriptor loss.
    """
    validate_training: bool = False
    include_mask_loss: bool = True
    nerf_loss: bool = False

@dataclass
class export_pseudo_labels_split:
    """Export pseudo labels on train, validation or test split.

    Args:
        enable_Homography_Adaptation: Enable homography adaptation duing export.
        split: The split to export pseudo labels on.
    """
    enable_Homography_Adaptation: bool = True
    split: Literal["training", "validation", "test"] = "training"


@tyro.conf.configure(tyro.conf.FlagConversionOff)
class main():
    """main class, script backbone.
    
    Args:
        config_path: Path to configuration.
        task: The task to be performed.
    """
    def __init__(self,
                 config_path: str,
                 task: Literal["train",
                               "export_pseudo_labels",
                               "HPatches_reliability",
                               "Hpatches_descriptors_evaluation"],
                training:options,
                pseudo_labels:export_pseudo_labels_split) -> None:

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f) 
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        if task == "train":

            self.model = get_model(self.config["model"], device=self.device)
            
            self.dataloader = get_loader(self.config, task, device=self.device, validate_training=training.validate_training)

            self.mask_loss = training.include_mask_loss
            self.nerf_loss = training.nerf_loss

            if self.config["pretraiend"]:
                
                model_state_dict =  self.model.state_dict()
                
                pretrained_dict = torch.load(f'{CKPT_PATH}\{self.config["pretraiend"]}', map_location=self.device)
                pretrained_state = pretrained_dict["model_state_dict"]
                
                for k,v in pretrained_state.items():
                    if k in model_state_dict.keys():
                        model_state_dict[k] = v
                
                self.model.load_state_dict(model_state_dict)
                print(f'\033[92m✅ Loaded pretrained model \033[0m')

                
                self.iteration = pretrained_dict["iteration"]

            self.train()


        if task == "export_pseudo_labels":

            self.pseudo_split = pseudo_labels.split
            self.enable_Homography_Adaptation = pseudo_labels.enable_Homography_Adaptation

            self.model = get_model(self.config["model"], device=self.device)
            self.dataloader = get_loader(self.config, task, device=self.device, export_split=self.pseudo_split)

            assert self.config["pretraiend"], "Use pretrained model to export pseudo labels."
            
            model_state_dict =  self.model.state_dict()
            
            pretrained_dict = torch.load(f'{CKPT_PATH}\{self.config["pretraiend"]}', map_location=self.device)
            pretrained_state = pretrained_dict["model_state_dict"]

            for k,v in pretrained_state.items():
                if k in model_state_dict.keys():
                    model_state_dict[k] = v
            
            self.model.load_state_dict(model_state_dict)
            print(f'\033[92m✅ Loaded pretrained model \033[0m')

            
            self.export_pseudo_labels()

    

    def train(self):

        if self.config["continue_training"]:
            iteration = self.iteration
        else:
            iteration = 0
        
        train_val(self.config, self.model, 
                  self.dataloader["train"], self.dataloader["validation"], 
                  self.mask_loss, iteration,
                  self.nerf_loss ,self.device)
    


    def export_pseudo_labels(self):
        
        ExportDetections(self.config, self.model, self.dataloader, self.pseudo_split, self.enable_Homography_Adaptation, self.device)
        

if __name__ == '__main__':
    tyro.cli(main)