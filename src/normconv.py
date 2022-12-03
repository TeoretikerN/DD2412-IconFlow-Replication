import torch
import torch.nn as nn


class NormConv(nn.Conv2d):
    def __init__(self, out_dims):
        super(NormConv, self).__init__(in_channels=1,
                                       out_channels=out_dims,
                                       kernel_size=2,
                                       padding='same',
                                       padding_mode='replicate',
                                       bias=False)
        
    def forward(self, x):
        x0, x1, x2, x3 = x.shape
        x = x.reshape((-1, x2, x3)).unsqueeze(dim=1)
        
        weight = self.weight - self.weight.mean([-2,-1])
        out = self._conv_forward(x, weight, self.bias)
        
        out = out.abs().reshape(x0, x1, x2, x3).mean(dim=1)
        
        return out
    
