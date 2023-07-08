import torch.nn as nn


class VGG_Block(nn.Module):
    def __init__(self, input, output, kn_size=3, pad=1, batch_norm=True, activation=True, maxpool=False):
        super().__init__()

        self.conv2d = nn.Conv2d(input,output,kernel_size=kn_size, stride=1, padding=pad)
        
        if activation:
            self.relu = nn.ReLU(inplace=True)
        
        if batch_norm:
            self.norm = nn.BatchNorm2d(output)
        
        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    
    def forward(self,x):

        x = self.conv2d(x)
        
        if hasattr(self, 'relu'):
            x = self.relu(x)
        
        if hasattr(self, 'norm'):
            x = self.norm(x)
        
        if hasattr(self, 'maxpool'):
            x = self.maxpool(x) 
        
        return x



class VGG_BACKBONE(nn.Module):
    def __init__(self,config):
        super(VGG_BACKBONE,self).__init__()

        self.block_1 = VGG_Block(1, config[0])
        
        self.block_2 = VGG_Block(config[0], config[1],maxpool=True)

        self.block_3 = VGG_Block(config[1], config[2])
        
        self.block_4 = VGG_Block(config[2], config[3],maxpool=True)

        self.block_5 = VGG_Block(config[3], config[4])
        
        self.block_6 = VGG_Block(config[4], config[5],maxpool=True)

        self.block_7 = VGG_Block(config[5], config[6])
        
        self.block_8 = VGG_Block(config[6], config[7])

    def forward(self,x):
        
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.block_7(x)
        x = self.block_8(x)

        return x