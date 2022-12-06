import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torchvision.models import resnet18, resnet50
from .unet import Unet
from .normconv import NormConv


class Colorizer(pl.LightningModule):
    def __init__(self,
                 contour_dim=1,
                 image_dim=3,
                 norm_dim=16,
                 embedding_dim=16,
                 style_dim=48,
                 decoder_dim=32,
                 decoder_depth=4,
                 toy_model=False,
                 lr=1e-4):
        super(Colorizer, self).__init__()
        self.contour_dim = contour_dim
        self.image_dim = image_dim
        self.norm_dim = norm_dim
        self.embedding_dim = embedding_dim
        self.style_dim = style_dim
        self.decoder_dim = decoder_dim
        self.decoder_depth = decoder_depth
        self.resnet = resnet18() if toy_model else resnet50()
        self.lr = lr
        
        self.contour_encoder = nn.Sequential(
            Unet(self.contour_dim,
                 self.embedding_dim,
                 depth=5,
                 n_filters=64),
            nn.Tanh()
        )
        
        self.contour_extractor = nn.Sequential(
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
        # TODO: Force in range (-.5, .5), divide by 2?
        return decoder_list
    
    def forward(self, contour, image):
        enc_c = self.contour_encoder(contour)
        enc_s = self.style_encoder(image)
        enc_s_cc = enc_s.roll(1, 0).contiguous()
        
        c0, c1 = enc_c.shape
        s0, s1, s2, s3 = enc_s.shape
        enc_c = enc_c.unsqueeze(-1).unsqueeze(-1).expand([c0,c1,s2,s3])
        z = torch.cat([enc_c, enc_s], 1)
        z = torch.transpose(z,1,3)
        z = torch.transpose(z,1,2)
        
        z_cc = torch.cat([enc_c, enc_s_cc], 1)
        z_cc = torch.transpose(z_cc,1,3)
        z_cc = torch.transpose(z_cc,1,2)
        
        out = self.decoder(z)
        out = torch.transpose(out,1,3)
        out = torch.transpose(out,2,3)
        
        out_cc = self.decoder(z_cc)
        out_cc = torch.transpose(out_cc,1,3)
        out_cc = torch.transpose(out_cc,2,3)
        return out, out_cc
    
    def countour_extract(self, image):
        return self.contour_extractor(image)
    
    def training_step(self, batch, batch_idx):
        loss = 0
        image, contour = batch
        colorized, colorized_cc = self(contour, image)
        extracted_contour = self.contour_extractor(image)
        
        # Get contour for consistency criterion (not pushing gradients)
        self.contour_extractor.requires_grad_(False)
        extracted_contour_cc = self.contour_extractor(colorized_cc)
        self.contour_extractor.requires_grad_(True)
        
        # MSE
        rec_loss = F.mse_loss(colorized, image)
        
        # Contour (TODO: Check shape, might need stack + mean)
        contour_loss = F.mse_loss(extracted_contour, contour)
        
        # Consistency criterion (TODO: Check shape, might need stack + mean)
        consistency_loss = F.mse_loss(extracted_contour_cc, contour)
        
        loss = rec_loss + contour_loss + consistency_loss
        
        # Logging to tensorboard
        self.log('Rec_Loss', rec_loss)
        self.log('Contour_Loss', contour_loss)
        self.log('Consistency_Loss', consistency_loss)
        self.log('Loss', loss)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
