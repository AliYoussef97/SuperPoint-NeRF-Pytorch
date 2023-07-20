# This code is from Superpoint[https://github.com/rpautrat/SuperPoint

import torch
import torchvision

def box_nms(prob, size, iou=0.1, min_prob=0.01, keep_top_k=0):

    pts = torch.nonzero(prob >= min_prob, as_tuple=False).to(torch.float32)

    size = torch.tensor(size/2.)

    boxes = torch.cat((pts - size, pts + size), dim=1).to(torch.float32)

    scores = prob[pts[:, 0].to(torch.int64), pts[:, 1].to(torch.int64)]

    indices = torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold=iou)

    pts = torch.index_select(pts,0,indices) #pts[indices, :]
    
    scores = torch.index_select(scores,0,indices) #scores[indices]

    if keep_top_k:
        k = min(scores.shape[0], keep_top_k)
        scores, indices = torch.topk(scores, k)
        pts = torch.index_select(pts, 0, indices) #pts[indices, :]
       
    
    nms_prob = torch.zeros_like(prob)
    nms_prob[pts[:, 0].to(torch.int64), pts[:, 1].to(torch.int64)] = scores
    
    return nms_prob