import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet18, resnet50
from unet import Unet
from normconv import NormConv


class Colorizer(nn.Module):
    def __init__(self,
                 contour_dim=1,
                 image_dim=3,
                 norm_dim=16,
                 embedding_dim=16,
                 style_dim=48,
                 decoder_dim=32,
                 decoder_depth=4,
                 toy_model=False):
        super(Colorizer, self).__init__()
        self.contour_dim = contour_dim
        self.image_dim = image_dim
        self.norm_dim = norm_dim
        self.embedding_dim = embedding_dim
        self.style_dim = style_dim
        self.decoder_dim = decoder_dim
        self.decoder_depth = decoder_depth
        self.resnet = resnet18() if toy_model else resnet50()
        
        self.image_encoder = nn.Sequential(
            Unet(self.contour_dim,
                 self.embedding_dim,
                 depth=5,
                 n_filters=64),
            nn.Tanh()
        )
        
        self.image_extractor = nn.Sequential(
            NormConv(self.norm_dim),
            Unet(self.norm_dim,
                 self.contour_dim,
                 depth=3,
                 n_filters=32))
        
        self.style_encoder = nn.Sequential(
            self.resnet,
            nn.Tanh()
        )
        
        self.decoder_list = self.add_decoder_layer([], self.embedding_dim + self.style_dim)
        for _ in range(self.decoder_depth - 2):
            self.decoder_list = self.add_decoder_layer(self.decoder_list, self.decoder_dim)
        self.decoder_list.append(nn.Linear(self.decoder_dim, self.image_dim))
        self.decoder_list.append(nn.Tanh())
                
        self.decoder = nn.Sequential(*self.decoder_list)
    
    def add_decoder_layer(self, decoder_list, dim_in):
        decoder_list.append(nn.Linear(dim_in, self.decoder_dim))
        decoder_list.append(nn.LayerNorm(self.decoder_dim))
        decoder_list.append(nn.ReLU())
        return decoder_list
    
    def forward(self, contour, image):
        enc_c = self.image_encoder(contour)
        enc_s = self.style_encoder(image)
        
        c0, c1 = enc_c.shape
        s0, s1, s2, s3 = enc_s.shape
        enc_c = enc_c.unsqueeze(-1).unsqueeze(-1).expand([c0,c1,s2,s3])
        z = torch.cat([enc_c, enc_s], 1)
        z = torch.transpose(z,1,3)
        z = torch.transpose(z,1,2)
        
        out = self.decoder(z)
        out = torch.transpose(z,1,3)
        out = torch.transpose(z,2,3)
        return out
