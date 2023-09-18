import yaml
import numpy as np
import torch
import tyro
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from superpoint.settings import CKPT_PATH
from superpoint.utils.data_loaders import get_loader
from superpoint.utils.get_model import get_model
import superpoint.evaluations.descriptor_evaluation as ev
from superpoint.utils.train_utils import move_to_device


@dataclass
class options:
    """Training options.

    Args:
        validate_training: configuation path
    """
    config_path:str
    alteration: str = "v"


@torch.no_grad()
def on_the_fly_HP_estimation(config,model,dataloader,device):
    correct=[]
    MS = []
    for _,data in enumerate(tqdm(dataloader)):
        
        data = move_to_device(data,device)
        
        output = model(data["image"])
        warped_output = model(data["warped_image"])

        output_nms = output["detector_output"]["prob_heatmap_nms"]
        warped_output_nms = warped_output["detector_output"]["prob_heatmap_nms"]

        output_desc = output["descriptor_output"]["desc"]
        warped_output_desc = warped_output["descriptor_output"]["desc"]

        out = {"image": data["image"].squeeze().cpu().numpy(),
                "warped_image": data["warped_image"].squeeze().cpu().numpy(),
                "prob": output_nms.squeeze().cpu().numpy(),
                "warped_prob": warped_output_nms.squeeze().cpu().numpy(),
                "desc": output_desc.squeeze().cpu().numpy().transpose(1,2,0),
                "warped_desc": warped_output_desc.squeeze().cpu().numpy().transpose(1,2,0),
                "homography": data["homography"].squeeze().cpu().numpy()}
        
        eval_baseline = ev.compute_homography(out, keep_k_points=config["model"]["detector_head"]["top_k"], 
                                              correctness_thresh=config["data"]["correctness_thresh"], 
                                              orb=False)
        correct.append(eval_baseline["correctness"])
        MS.append(eval_baseline["matching_score"])

    correct = np.mean(correct)
    MS = np.mean(MS)
    print("Correctness:",correct,"MS:",MS)


if __name__ == "__main__":
    args = tyro.cli(options)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f) 

    model = get_model(config["model"], device=device)

    model_state_dict =  model.state_dict()
    
    pretrained_dict = torch.load(Path(CKPT_PATH,config["pretrained"]), map_location=device)
    pretrained_state = pretrained_dict["model_state_dict"]
    
    for k,v in pretrained_state.items():
        if k in model_state_dict.keys():
            model_state_dict[k] = v
    
    model.load_state_dict(model_state_dict)
    model.eval()

    config["data"]["alteration"] = args.alteration
    dataloader = get_loader(config, "export_HPatches_Descriptors", device="cpu", validate_training=False)

    on_the_fly_HP_estimation(config,model,dataloader,device)