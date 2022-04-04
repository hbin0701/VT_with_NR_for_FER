import timm
import torch.nn as nn
import torch
from .Rescale import ResizingNetwork
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
from datetime import datetime
from .STN import STN

class VanillaSwinT(nn.Module):
    def __init__(self, n_classes:int, size:str="small"):
        super(VanillaSwinT, self).__init__()

        self.model = timm.create_model("swin_base_patch4_window7_224", pretrained=True)
        print(self.model)
        self.model.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 8), 
        )
        self.stn = STN()        
        self.res = ResizingNetwork()
        
    def forward(self, x):
        x = self.stn(x)
        x = self.res(x)
        x = self.model(x)  
        return x
    
