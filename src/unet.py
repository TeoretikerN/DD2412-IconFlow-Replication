import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock

## Building_blocks
# Helper functions
def get_down_conv(in_channels, out_channels, kernels, stride=1, padding=0, bias=False):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=kernels,
                     stride=stride,
                     padding=padding,
                     bias=bias)

def get_up_conv(in_channels, out_channels, scale=2, mode='nearest'):
    # Iconflow repo uses "nearest" mode (not mentioned in paper),
    # "bilinear" might be too costly.
    return nn.Sequential(nn.Upsample(mode=mode,
                                     scale_factor=scale),
                         get_down_conv(in_channels,
                                       out_channels,
                                       1)
                         )
    
def autocrop(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor):
    """
    Source : https://github.com/ELEKTRONN/elektronn3/blob/master/elektronn3/models/unet.py
    Max Planck Institute of Neurobiology, Munich, Germany
    Author: Martin Drawitsch
    
    Center-crops the encoder_layer to the size of the decoder_layer,
    so that merging (concatenation) between levels/blocks is possible.
    This is only necessary for input sizes != 2**n for 'same' padding and always required for 'valid' padding.
    """
    if encoder_layer.shape[2:] != decoder_layer.shape[2:]:
        ds = encoder_layer.shape[2:]
        es = decoder_layer.shape[2:]
        assert ds[0] >= es[0]
        assert ds[1] >= es[1]
        if encoder_layer.dim() == 4:  # 2D
            encoder_layer = encoder_layer[
                            :,
                            :,
                            ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                            ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2)
                            ]
        elif encoder_layer.dim() == 5:  # 3D
            assert ds[2] >= es[2]
            encoder_layer = encoder_layer[
                            :,
                            :,
                            ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                            ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2),
                            ((ds[2] - es[2]) // 2):((ds[2] + es[2]) // 2),
                            ]
    return encoder_layer, decoder_layer


# Up and Down blocks
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True, bias=True, padding=1):
        super(DownBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.padding = padding

        if pool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        else:
            self.pool = None

        self.conv1 = get_down_conv(self.in_channels,
                                   self.out_channels,
                                   3,
                                   padding=padding,
                                   bias=bias)
        
        self.conv2 = get_down_conv(self.out_channels,
                                   self.out_channels,
                                   3,
                                   padding=padding,
                                   bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        skip_conn = x
        if self.pool is not None:
            x = self.pool(x)
        return x, skip_conn


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, padding=1):
        super(UpBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.padding = padding

        self.conv1 = get_up_conv(self.in_channels,
                              self.out_channels)

        self.conv2 = get_down_conv(2*self.out_channels,
                                   self.out_channels,
                                   3,
                                   padding=padding,
                                   bias=bias)
        
        self.conv3 = get_down_conv(self.out_channels,
                            self.out_channels,
                            3,
                            padding=padding,
                            bias=bias)
        
    def forward(self, x_encoded, x_decoded):
        x_decoded = self.conv1(x_decoded)
        x_cropped_encoded, _ = autocrop(x_encoded, x_decoded)
        x = torch.cat((x_cropped_encoded, x_decoded), dim=1)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        return x
    

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, depth, n_filters, out_kernels = 3, norm_layer=None):
        super(Unet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.initial_filters = n_filters
        self.out_kernels = out_kernels
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.output_conv = None
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        n_in = self.in_channels
        for block in range(depth):
            n_out = int(self.initial_filters*(2**block))
            
            pool = depth-block > 1
            if n_out == n_in:
                self.encoder_blocks.append(BasicBlock(n_in, n_out, downsample=nn.Identity(),norm_layer=norm_layer))
            else:
                self.encoder_blocks.append(DownBlock(n_in,
                                                     n_out,
                                                     pool=pool))
            n_in = n_out
        
        for block in range(depth-1):
            n_out = n_in // 2
            self.decoder_blocks.append(UpBlock(n_in,
                                               n_out))
            n_in = n_out
        
        self.output_conv = nn.ConvTranspose2d(n_in,
                                              self.out_channels,
                                              self.out_kernels,
                                              padding=1)
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                try:
                  nn.init.constant_(module.bias, 0)
                  nn.init.xavier_normal_(module.weight)
                except AttributeError: # Bias = False
                  nn.init.xavier_normal_(module.weight)

    def forward(self, x):
        skip_conns = [0]*self.depth
        for i, block in enumerate(self.encoder_blocks):
            x, skip_conns[i] = block(x)
        
        for i, block in enumerate(self.decoder_blocks):
            x = block(skip_conns[-2-i], x)
        x = self.output_conv(x)
        return x

if __name__ == "__main__":
    """
    Testing
    """
    import numpy as np
    from torch.autograd import Variable
    from torchinfo import summary
    model = Unet(1, 16, 5, 64)
    print(summary(model, input_size=(5, 1, 128, 128)))
    x = Variable(torch.FloatTensor(np.random.random((5, 1, 128, 128))))
    out = model(x)
    loss = torch.sum(out)
    loss.backward()
