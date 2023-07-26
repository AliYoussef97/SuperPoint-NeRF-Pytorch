import torch

def filter_points(points, shape, device='cpu', return_mask=False):
    """
    Remove points close to the border of the image.
    input:
        points: (N,2)
        shape: torch tensor (H,W)
    output:
        points: (N,2)
    """
    if len(points)!=0:
        mask = (points >= 0) & (points <= torch.tensor(shape,dtype=torch.float32,device=device)-1)
        mask = torch.logical_and(mask[:,0], mask[:,1])
        if return_mask:
            return points[mask], mask    
        return points[mask]
    else:
        return points

def compute_keypoint_map(points, shape, device='cpu'):
    """
    input:
        points: (N,2)
        shape: torch tensor (H,W)
    output:
        kmap: (H,W)
    """
    coord = torch.minimum(torch.round(points).to(torch.int32), 
                          torch.tensor(shape,dtype=torch.int32,device=device)-1)
    kmap = torch.zeros((shape), dtype=torch.int32, device=device)   
    kmap[coord[:,0],coord[:,1]] = 1
    return kmap


def warp_points(points, homography, device='cpu'):
    """
    :param points: (N,2), tensor
    :param homographies: [B, 3, 3], batch of homographies
    :return: warped points B,N,2
    """
    if len(points.shape)==0:
        return points

    points = torch.fliplr(points)
    
    batch_size = homography.shape[0]
    
    points = torch.cat((points, torch.ones((points.shape[0], 1),device=device)),dim=1)

    warped_points = torch.tensordot(homography, points.transpose(1,0),dims=([2], [0]))

    warped_points = warped_points.reshape([batch_size, 3, -1])
    
    warped_points = warped_points.transpose(2, 1)
    
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    
    warped_points = torch.flip(warped_points,dims=(2,))
    
    warped_points = warped_points.squeeze(0)
       
    return warped_points