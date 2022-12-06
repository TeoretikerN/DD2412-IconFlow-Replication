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
        
        # print("Weight shape:", self.weight.shape)
        weight = self.weight - self.weight.mean(dim=[-2,-1], keepdim=True)
        # print("Weight shape:", weight.shape)
        out = self._conv_forward(x, weight, self.bias)
        out = out.reshape(x0, x1, weight.shape[0], x2, x3)
        # print("out shape:", out.shape)
        out = out.abs().mean(dim=1)
        # print("out shape:", out.shape)
        return out
    

if __name__ == "__main__":
    """
    Testing
    """
    import numpy as np
    from torch.autograd import Variable
    #from torchinfo import summary
    model = NormConv(16)
    dims = (32, 16, 64, 64)
    #print(summary(model, input_size=dims))
    x = Variable(torch.FloatTensor(np.random.random(dims)))
    out = model(x)
    loss = torch.sum(out)
    loss.backward()
