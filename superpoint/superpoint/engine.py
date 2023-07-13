import tyro
import yaml
import torch
from typing import Literal
from superpoint.settings import EXPER_PATH
from superpoint.utils.get_model import get_model
from superpoint.utils.data_loaders import get_loader
from superpoint.engine_solvers.train import train_val



class main():
    def __init__(self,
                 config_path: str,
                 task: Literal["train",
                               "test",
                               "export_detections",
                               "HPatches_reliability",
                               "Hpatches_descriptors_evaluation"]) -> None:

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f) 
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = get_model(self.config["model"], device=self.device)
        self.dataloader = get_loader(self.config, task, device=self.device)
    
        if self.config["pretraiend"]:
            
            model_state_dict =  self.model.state_dict()
            
            pretrained_dict = torch.load(f'{EXPER_PATH}\{self.config["pretraiend"]}')
            pretrained_dict = pretrained_dict["model_state_dict"]
            
            for k,v in pretrained_dict.items():
                if k in model_state_dict.keys():
                    model_state_dict[k] = v
            
            self.model.load_state_dict(model_state_dict)
            
            if self.config["continue_traning"]:
                self.iteration = pretrained_dict["iteration"]
            else:
                self.iteration = 0
        
        if task == "train":
            getattr(self,task)(self.config, self.model, self.dataloader["train"], self.dataloader["validation"], self.iteration, self.device)
    

    def train(self, config, model, train_loader, validation_loader, iteration, device):

        train_val(config, model, train_loader, validation_loader, iteration, device)
        

if __name__ == '__main__':
    tyro.cli(main)