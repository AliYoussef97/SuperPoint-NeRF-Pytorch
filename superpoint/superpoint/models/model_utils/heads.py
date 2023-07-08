import torch.nn as nn
from superpoint.models.model_utils.VGG_Backbone import VGG_Block
from superpoint.models.model_utils.sp_utils import box_nms
import torch



class Detector_head(nn.Module):
    def __init__(self, config):
        super(Detector_head,self).__init__()
        self.config = config

        self.convPa = VGG_Block(config["detector_dim"][0],config["detector_dim"][1], activation=True)
        
        self.convPb = VGG_Block(config["detector_dim"][1], pow(config["grid_size"], 2)+1, kn_size=1, pad=0, activation=False)

        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        
        output = {}
        
        x = self.convPa(x)
        x = self.convPb(x) # raw output -> (B,grid_size**2+1,H,W)
        output.setdefault("logits", x)


        x_prob = self.softmax(x) # probability -> (B,grid_size**2+1,H,W)
        x_prob = x[:,:-1,:,:] # Dustbin removal (B,grid_size**2+1,H,W) -> (B,grid_size**2,H,W)
        x_prob = nn.functional.pixel_shuffle(x_prob, self.config["grid_size"]) # (B,grid_size**2,H,W) -> (B,1,H*grid_size,W*grid_size)
        x_prob = x_prob.squeeze(1) # Remove channel dimension -> (B,H*grid_size,W*grid_size)
        output.setdefault("prob", x_prob)

        if self.config["nms"]:
            
            x_prob = [box_nms(prob=prob, # (H*grid_size,W*grid_size)
                              size=self.config["nms"],
                              keep_top_k=self.config["top_k"]) for prob in x_prob]
            
            x_prob = torch.stack(x_prob) # (B,H*grid_size,W*grid_size)
            output.setdefault("prob_nms", x_prob)

        pred = torch.ge(x_prob,self.config["det_thresh"]).to(torch.int32) # (B,H*grid_size,W*grid_size)
        
        output.setdefault("pred", pred)

        return output



class Descriptor_head(nn.Module):
    def __init__(self, config):
        super(Descriptor_head,self).__init__()
        self.config = config

        self.convDa = VGG_Block(config["descriptor_dim"][0],config["descriptor_dim"][1], activation=True)
        
        self.convDb = VGG_Block(config["descriptor_dim"][1],config["descriptor_dim"][1] , kn_size=1, pad=0, activation=False)

    def forward(self,x):
        
        output = {}
        
        x = self.convDa(x)
        x = self.convDb(x)
        output.setdefault("desc_raw", x)   

        desc = nn.functional.interpolate(x, scale_factor=self.config["grid_size"], mode='bicubic', align_corners=False) #(B,256,H,W)
        desc = nn.functional.normalize(x, p=2, dim=1) #(B,256,H,W)
        output.setdefault("desc", desc)

        return output