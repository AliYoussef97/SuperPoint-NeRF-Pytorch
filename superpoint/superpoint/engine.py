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
class validate_training:
    validate_training: bool = False

@dataclass
class export_pseudo_labels_split:
    split: Literal["training", "validation", "test"] = "training"


@tyro.conf.configure(tyro.conf.FlagConversionOff)
class main():
    """main class, script backbone.
    
    Args:
        config_path: Path to configuration.
        task: The task to be performed.
        training: Whether to validate during training or not.
        pseudo_labels: Export pseudo labels on train, validation or test split.
    """
    def __init__(self,
                 config_path: str,
                 task: Literal["train",
                               "export_pseudo_labels",
                               "HPatches_reliability",
                               "Hpatches_descriptors_evaluation"],
                training:validate_training,
                pseudo_labels:export_pseudo_labels_split) -> None:

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f) 
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        if task == "train":

            self.model = get_model(self.config["model"], device=self.device)
            
            self.dataloader = get_loader(self.config, task, device=self.device, validate_training=training.validate_training)

            if self.config["pretraiend"]:
                
                model_state_dict =  self.model.state_dict()
                
                pretrained_dict = torch.load(f'{CKPT_PATH}\{self.config["pretraiend"]}', map_location=self.device)
                pretrained_state = pretrained_dict["model_state_dict"]
                
                for k,v in pretrained_state.items():
                    if k in model_state_dict.keys():
                        model_state_dict[k] = v
                
                self.model.load_state_dict(model_state_dict)
                
                self.iteration = pretrained_dict["iteration"]

            self.train()


        if task == "export_pseudo_labels":

            self.pseudo_split = pseudo_labels.split

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
            
            self.export_pseudo_labels()

    

    def train(self):

        if self.config["continue_training"]:
            iteration = self.iteration
        else:
            iteration = 0
        
        train_val(self.config, self.model, self.dataloader["train"], self.dataloader["validation"], iteration, self.device)
    


    def export_pseudo_labels(self):
        
        ExportDetections(self.config, self.model, self.dataloader, self.pseudo_split, self.device)
        

if __name__ == '__main__':
    tyro.cli(main)