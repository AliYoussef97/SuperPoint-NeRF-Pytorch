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
from superpoint.utils.train_utils import move_to_device
torch.set_grad_enabled(False)

@dataclass
class options:
    """Training options.

    Args:
        validate_training: configuation path
    """
    config_path:str
    alteration: str = "v"


def warp_keypoints(keypoints, H):
    num_points = keypoints.shape[0]
    homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))],
                                        axis=1)
    warped_points = np.dot(homogeneous_points, np.transpose(H))
    return warped_points[:, :2] / warped_points[:, 2:]

def filter_keypoints(points, shape):
    """ Keep only the points whose coordinates are
    inside the dimensions of shape. """
    mask = (points[:, 0] >= 0) & (points[:, 0] < shape[0]) &\
            (points[:, 1] >= 0) & (points[:, 1] < shape[1])
    return points[mask, :]

def keep_true_keypoints(points, H, shape):
    """ Keep only the points whose warped coordinates by H
    are still inside shape. """
    warped_points = warp_keypoints(points[:, [1, 0]], H)
    warped_points[:, [0, 1]] = warped_points[:, [1, 0]]
    mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[0]) &\
            (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[1])
    return points[mask, :]

def select_k_best(points, k):
    """ Select the k most probable points (and strip their proba).
    points has shape (num_points, 3) where the last coordinate is the proba. """
    sorted_prob = points[points[:, 2].argsort(), :2]
    start = min(k, points.shape[0])
    return sorted_prob[-start:, :]



@torch.no_grad()
def on_the_fly_HP_estimation(config,model,dataloader,device):

    repeatability=[]
    LE_error =[]
    N1s=[]
    N2s=[]
    for _, d in enumerate(tqdm(dataloader)):
        
        d = move_to_device(d,device)
        
        output = model(d["image"])
        warped_output = model(d["warped_image"])

        output_nms = output["detector_output"]["prob_heatmap_nms"]
        warped_output_nms = warped_output["detector_output"]["prob_heatmap_nms"]


        data = {"image": d["image"].squeeze().cpu().numpy(),
                "warped_image": d["warped_image"].squeeze().cpu().numpy(),
                "prob": output_nms.squeeze().cpu().numpy(),
                "warped_prob": warped_output_nms.squeeze().cpu().numpy(),
                "homography": d["homography"].squeeze().cpu().numpy()}
        

        shape = data['warped_prob'].shape
        H = data['homography']

        keypoints = np.where(data['prob'] > 0)
        prob = data['prob'][keypoints[0], keypoints[1]]
        keypoints = np.stack([keypoints[0], keypoints[1]], axis=-1)
        warped_keypoints = np.where(data['warped_prob'] > 0)
        warped_prob = data['warped_prob'][warped_keypoints[0], warped_keypoints[1]]
        warped_keypoints = np.stack([warped_keypoints[0],
                                    warped_keypoints[1],
                                    warped_prob], axis=-1)
        warped_keypoints = keep_true_keypoints(warped_keypoints, np.linalg.inv(H),
                                            data['prob'].shape)  
        
        true_warped_keypoints = warp_keypoints(keypoints[:, [1, 0]], H)
        true_warped_keypoints = np.stack([true_warped_keypoints[:, 1],
                                            true_warped_keypoints[:, 0],
                                            prob], axis=-1)
        true_warped_keypoints = filter_keypoints(true_warped_keypoints, shape)

        warped_keypoints = select_k_best(warped_keypoints, config["model"]["detector_head"]["top_k"])
        true_warped_keypoints = select_k_best(true_warped_keypoints, config["model"]["detector_head"]["top_k"])

        N1 = true_warped_keypoints.shape[0]
        N2 = warped_keypoints.shape[0]
        N1s.append(N1)
        N2s.append(N2)
        true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1)
        warped_keypoints = np.expand_dims(warped_keypoints, 0)

        norm = np.linalg.norm(true_warped_keypoints - warped_keypoints,
                                ord=None, axis=2)
        count1 = 0
        count2 = 0
        LE_1 = None
        LE_2 = None
        if N2 != 0:
            min1 = np.min(norm, axis=1)
            count1 = np.sum(min1 <= config["data"]["correctness_thresh"])
            LE_1 = min1[min1 <= config["data"]["correctness_thresh"]]
        if N1 != 0:
            min2 = np.min(norm, axis=0)
            count2 = np.sum(min2 <= config["data"]["correctness_thresh"])
            LE_1 = min2[min2 <= config["data"]["correctness_thresh"]]

        if N1 + N2 > 0:
            LE=0
            repeatability.append((count1 + count2) / (N1 + N2))
            if LE_1 is not None:
                LE +=(LE_1.sum())/(count1 + count2)
            if LE_2 is not None:
                LE +=(LE_2.sum())/(count1 + count2)
            LE_error.append(LE)                

    repeatability = np.mean(repeatability)
    LE = np.mean(LE_error)
    print(f"repeatability:{repeatability},Local_Err:{LE}")


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
    dataloader = get_loader(config, "export_HPatches_Repeatability", device="cpu", validate_training=False)

    on_the_fly_HP_estimation(config,model,dataloader,device)