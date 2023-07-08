import torch.nn as nn
from superpoint.models.model_utils.VGG_Backbone import VGG_BACKBONE
from superpoint.models.model_utils.heads import Detector_head, Descriptor_head

class SuperPoint(nn.Module):
    def __init__(self, config):
        super(SuperPoint,self).__init__()
        self.config = config

        self.backbone = VGG_BACKBONE(config["vgg_cn"])

        self.detector_head = Detector_head(config["detector_head"])

        if config["name"] == "superpoint":
            self.descriptor_head = Descriptor_head(config["descriptor_head"])
    
    def forward(self,x):
        
        output = {}
        
        feature_map = self.backbone(x)

        detector_output = self.detector_head(feature_map)
        output.setdefault("detector_output", detector_output)

        if hasattr(self, 'descriptor_head'):
            descriptor_output = self.descriptor_head(feature_map)
            output.setdefault("descriptor_output", descriptor_output)
        
        return output