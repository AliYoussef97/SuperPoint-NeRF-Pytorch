# Parts of this code is from SuperGlue[https://github.com/magicleap/SuperGluePretrainedNetwork]

from superpoint.settings import DATA_PATH,CKPT_PATH
from superpoint.utils.get_model import get_model
import torch
import numpy as np
from dataclasses import dataclass 
import tyro
import yaml
from pathlib import Path
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
torch.set_grad_enabled(False)

@tyro.conf.configure(tyro.conf.FlagConversionOff)
@dataclass
class options:
    """Training options.

    Args:
        validate_training: configuation path
    """
    config_path:str
    max_length: int = -1
    shuffle: bool = False


def keep_shared_points(keypoint_map, keep_k_points=1024):
    """
    Compute a list of keypoints from the map, filter the list of points by keeping
    only the points that once mapped by H are still inside the shape of the map
    and keep at most 'keep_k_points' keypoints in the image.
    """
    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]
   
    def remove_borders(keypoints, border: int, height: int, width: int):
        """ Removes keypoints too close to the border """
        mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
        mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
        mask = mask_h & mask_w
        return keypoints[mask]

    h,w = keypoint_map.shape
    keypoints = np.where(keypoint_map > 0)
    prob = keypoint_map[keypoints[0], keypoints[1]]
    keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)
    keypoints = remove_borders(keypoints,4,h,w)
    keypoints = select_k_best(keypoints, keep_k_points)

    return keypoints.astype(int)


def match(data, keep_k_points=1024):
    """
    Compute the homography between 2 sets of detections and descriptors inside data.
    """

    keypoints = keep_shared_points(data['prob'],
                                    keep_k_points)
    warped_keypoints = keep_shared_points(data['warped_prob'],
                                          keep_k_points)
    desc = data['desc'][keypoints[:, 0], keypoints[:, 1]]
    warped_desc = data['warped_desc'][warped_keypoints[:, 0],
                                      warped_keypoints[:, 1]]
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc, warped_desc)
    matches = sorted(matches,key=lambda x:(x.distance<0.25))
    matches_idx = np.array([m.queryIdx for m in matches])
    if len(matches_idx) == 0:
        return {'correctness': 0.,
                'keypoints1': keypoints,
                'keypoints2': warped_keypoints,
                'matches': [],
                'inliers': [],
                'homography': None}
    m_keypoints = keypoints[matches_idx, :]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_warped_keypoints = warped_keypoints[matches_idx, :]

    m_keypoints=m_keypoints[:, [1, 0]]
    m_warped_keypoints = m_warped_keypoints[:, [1, 0]]

    return m_keypoints, m_warped_keypoints, keypoints


def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:
        w_new, h_new = resize[0], resize[1]

    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def read_image(path, device, resize, rotation, resize_float):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales


def scale_intrinsics(K, scales):
    scales = np.diag([1./scales[0], 1./scales[1], 1.])
    return np.dot(scales, K)


def rotate_intrinsics(K, image_shape, rot):
    """image_shape is the shape of the image after rotation"""
    assert rot <= 3
    h, w = image_shape[:2][::-1 if (rot % 2) else 1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rot = rot % 4
    if rot == 1:
        return np.array([[fy, 0., cy],
                         [0., fx, w-1-cx],
                         [0., 0., 1.]], dtype=K.dtype)
    elif rot == 2:
        return np.array([[fx, 0., w-1-cx],
                         [0., fy, h-1-cy],
                         [0., 0., 1.]], dtype=K.dtype)
    else: 
        return np.array([[fy, 0., h-1-cy],
                         [0., fx, cx],
                         [0., 0., 1.]], dtype=K.dtype)


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array([[np.cos(r), -np.sin(r), 0., 0.],
                  [np.sin(r), np.cos(r), 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]], dtype=np.float32)
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)


def compute_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T
    p1Ep0 = np.sum(kpts1 * Ep0, -1)
    Etp1 = kpts1 @ E 
    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2)
                    + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))
    return d


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    assert E is not None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs


@torch.no_grad()
def estimate_pose_errors(config, model, pairs, device):
    
    pose_errors = []
    precisions = []
    matching_scores = []
    all_errors = []
    for i,pair in enumerate(tqdm(pairs)):
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        
        if len(pair) >= 5:
            rot0, rot1 = int(pair[2]), int(pair[3])
        else:
            rot0, rot1 = 0, 0

        image0, inp0, scales0 = read_image(
            Path(DATA_PATH,config["data"]["images_path"],name0), device, config["data"]["resize"], rot0, config["data"]["resize_float"])
        image1, inp1, scales1 = read_image(
            Path(DATA_PATH,config["data"]["images_path"],name1), device, config["data"]["resize"], rot1, config["data"]["resize_float"])

        
        output = model(inp0)
        output2 = model(inp1)
        output_nms = output["detector_output"]["prob_heatmap_nms"]
        warped_output_nms = output2["detector_output"]["prob_heatmap_nms"]
        output_desc = output["descriptor_output"]["desc"]
        warped_output_desc = output2["descriptor_output"]["desc"]
        
        out = {"image": image0,
                "warped_image": image1,
                "prob": output_nms.squeeze().cpu().numpy(),
                "warped_prob": warped_output_nms.squeeze().cpu().numpy(),
                "desc": output_desc.squeeze().cpu().numpy().transpose(1,2,0),
                "warped_desc": warped_output_desc.squeeze().cpu().numpy().transpose(1,2,0)}
        
        mkpts0, mkpts1, kpts0 = match(out,config["model"]["detector_head"]["top_k"])
        
        K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
        K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
        T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

        K0 = scale_intrinsics(K0, scales0)
        K1 = scale_intrinsics(K1, scales1)

        if rot0 != 0 or rot1 != 0:
            cam0_T_w = np.eye(4)
            cam1_T_w = T_0to1
            if rot0 != 0:
                K0 = rotate_intrinsics(K0, image0.shape, rot0)
                cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
            if rot1 != 0:
                K1 = rotate_intrinsics(K1, image1.shape, rot1)
                cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
            cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
            T_0to1 = cam1_T_cam0

        epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
        correct = epi_errs < config["data"]["epi_thrsehold"]
        num_correct = np.sum(correct)
        precision = np.mean(correct) if len(correct) > 0 else 0
        matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

        thresh = 1.  # In pixels relative to resized image size.
        ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
        if ret is None:
            err_t, err_R = np.inf, np.inf
        else:
            R, t, inliers = ret
            err_t, err_R = compute_pose_error(T_0to1, R, t)

        out_eval = {'error_t': err_t,
                    'error_R': err_R,
                    'precision': precision,
                    'matching_score': matching_score,
                    'num_correct': num_correct,
                    'epipolar_errors': epi_errs}
        all_errors.append(out_eval)

    for errors in all_errors:
        pose_error = np.maximum(errors['error_t'], errors['error_R'])
        pose_errors.append(pose_error)
        precisions.append(errors['precision'])
        matching_scores.append(errors['matching_score'])
    
    thresholds = [5, 10, 20]
    aucs = pose_auc(pose_errors, thresholds)
    aucs = [100.*yy for yy in aucs]
    prec = 100.*np.mean(precisions)
    ms = 100.*np.mean(matching_scores)
    print('Evaluation Results (mean over {} pairs):'.format(len(pairs)))
    print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
        aucs[0], aucs[1], aucs[2], prec, ms))



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
    model.eval().to(device)

    input_pairs = Path(DATA_PATH,config["data"]["gt_pairs"])
    
    with open(input_pairs, 'r') as f:
         pairs = [l.split() for l in f.readlines()]
    
    if args.shuffle:
        random.Random(0).shuffle(pairs)
    
    if args.max_length > -1:
        pairs = pairs[0:np.min([len(pairs), args.max_length])]
    
    estimate_pose_errors(config,model,pairs,device)
