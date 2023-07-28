import torch
import torch.nn.functional as F
from superpoint.data.data_utils.kp_utils import warp_points


def detector_loss(logits,
                  kpts_heatmap,
                  valid_mask, 
                  grid_size=8, 
                  include_mask=False,
                  device="cpu"):
    
    labels = kpts_heatmap.unsqueeze(1).to(torch.float32) # (B,1,H,W)
    labels = torch.pixel_unshuffle(labels, grid_size) # (B,1,H,W) -> (B,grid_size**2,H/grid_size,W/grid_size)
    
    B,_,Hc,Wc = labels.shape
    
    dustbin = torch.ones(size=[B,1,Hc,Wc], device=device) # (B,1,H/grid_size,W/grid_size)
    labels = torch.cat([2*labels, dustbin], dim=1) #(B,grid_size**2+1,H/grid_size,W/grid_size)
    
    random_tie_break = torch.distributions.uniform.Uniform(0,0.1).sample(labels.shape).to(device)
    labels = torch.argmax(labels+random_tie_break, dim=1) # (B,H/grid_size,W/grid_size)
    
    valid_mask = torch.ones_like(kpts_heatmap,device=device) if include_mask is False else valid_mask
    valid_mask = valid_mask.unsqueeze(1).to(torch.float32) # (B,1,H,W)
    valid_mask = torch.pixel_unshuffle(valid_mask, grid_size)
    valid_mask = torch.prod(valid_mask, dim=1) # (B,H/grid_size,W/grid_size)
    
    det_loss = F.cross_entropy(logits, labels, reduction='none') # (B,H/grid_size,W/grid_size)
    
    weigthted_det_loss = torch.divide(torch.sum(det_loss*valid_mask, dim=(1,2)),
                                      torch.sum(valid_mask, dim=(1,2))+1e-10)
    
    weigthted_det_loss = torch.mean(weigthted_det_loss)
    
    return weigthted_det_loss


def descriptor_loss(config,descriptors,warped_descriptors,homographies,valid_mask=None,device="cpu"):

    grid_size = config["descriptor_head"]["grid_size"]
    lambda_d = config["descriptor_head"]["lambda_d"]
    lambda_loss = config["descriptor_head"]["lambda_loss"]
    positive_margin = config["descriptor_head"]["positive_margin"]
    negative_margin = config["descriptor_head"]["negative_margin"]

    B, C, Hc, Wc = descriptors.shape
    coord_cells = torch.stack(torch.meshgrid([torch.arange(Hc), torch.arange(Wc)],indexing="ij"), dim=-1).to(device) # (H,W,2)

    coord_cells = coord_cells * grid_size + grid_size // 2
    coord_cells = coord_cells.reshape(-1,2)
    
    warped_coord_cells = warp_points(coord_cells, homographies, device=device)
    coord_cells = torch.reshape(coord_cells, [1,1,1,Hc,Wc,2]).type(torch.float32) # (1,1,1,H,W,2)
    warped_coord_cells = torch.reshape(warped_coord_cells, [B, Hc, Wc, 1, 1, 2]) # (B,H,W,1,1,2)

    cell_distances = torch.linalg.vector_norm(coord_cells - warped_coord_cells, ord=2, dim=-1) # (B,H,W,H,W)
    s = (cell_distances<=(grid_size-0.5)).to(torch.float32) # (B,H,W,H,W)

    descriptors = torch.reshape(descriptors, [B, -1, Hc, Wc, 1, 1])
    descriptors = F.normalize(descriptors, p=2, dim=1)

    warped_descriptors = torch.reshape(warped_descriptors, [B, -1, 1, 1, Hc, Wc])
    warped_descriptors = F.normalize(warped_descriptors, p=2, dim=1)

    desc_dot = torch.sum(descriptors * warped_descriptors, dim=1)

    desc_dot = F.relu(desc_dot)

    desc_dot = torch.reshape(F.normalize(torch.reshape(desc_dot, [B, Hc, Wc, Hc * Wc]),
                                                 p=2,
                                                 dim=3), [B, Hc, Wc, Hc, Wc])
    
    desc_dot = torch.reshape(F.normalize(torch.reshape(desc_dot, [B, Hc * Wc, Hc, Wc]),
                                                 p=2,
                                                 dim=1), [B, Hc, Wc, Hc, Wc])
    
    positive_margin = torch.maximum(torch.tensor(0.,device=device), positive_margin - desc_dot)
    negative_margin = torch.maximum(torch.tensor(0.,device=device), desc_dot - negative_margin)

    desc_loss = lambda_d * s * positive_margin + (1 - s) * negative_margin


    valid_mask = torch.ones([B, Hc*grid_size, Wc*grid_size],
                             dtype=torch.float32, device=device) if valid_mask is None else valid_mask
    valid_mask = valid_mask.unsqueeze(dim=1).type(torch.float32)
    valid_mask = torch.pixel_unshuffle(valid_mask, grid_size)
    valid_mask = torch.prod(valid_mask, dim=1).type(torch.float32)
    valid_mask = torch.reshape(valid_mask, [B, 1, 1, Hc, Wc])

    normalization = torch.sum(valid_mask)*(Hc*Wc)

    desc_loss = lambda_loss*torch.sum(valid_mask * desc_loss)/normalization

    return desc_loss