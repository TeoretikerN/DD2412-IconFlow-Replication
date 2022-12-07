import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.distributions import Normal
from IconFlow.iconflow.model.styleflow.flow import build_model


class NormalizingFlow(pl.LightningModule):
    
    def __init__(self,
                 colorizer,
                 width=512,
                 depth=4,
                 condition_size=2,
                 lr=1e-4):
        super(NormalizingFlow, self).__init__()
        self.colorizer = colorizer
        self.width = width
        self.depth = depth
        self.style_dim = colorizer.style_dim
        self.condition_size = condition_size
        self.lr = lr

        self.flow = build_model(
            self.style_dim,
            (self.width,) * self.depth,
            self.condition_size,
            1,
            True
        )

    def forward(self, style, location):
        batch_size = style.shape[0]
        zeros = torch.zeros(batch_size, 1).to(style)

        z, dlogp = self.flow(style, location, zeros)
        return z, dlogp

    def training_step(self, batch, batch_idx):
        loss = 0
        image, location = batch

        # Get style vector for training CNF (not pushing gradients)
        self.colorizer.style_encoder.requires_grad_(False)
        encoded_style = self.colorizer.style_encode(image)
        self.colorizer.style_encoder.requires_grad_(True)

        #print('location shape:', location.shape)
        #print('encoded shape:', encoded_style.shape)

        self.flow.train()
        z, dlogp = self(encoded_style, location)
        #print('z shape:', z.shape, 'dlogp shape:', dlogp.shape)

        logpz = Normal(0, 1).log_prob(z).sum(-1)
        logpx = logpz - dlogp
        nll = -logpx.mean()
        bpd = nll / self.style_dim / math.log(2)

        # Logging to tensorboard
        self.log('NLL', nll)
        self.log('BPD', bpd)

        return bpd

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    """
    Testing
    """
    import numpy as np
    from torch.autograd import Variable
    from .colorizer_model import Colorizer

    colorizer = Colorizer()
    flow = NormalizingFlow(colorizer)

    style = Variable(torch.FloatTensor(np.random.random((32, 48))))
    location = Variable(torch.FloatTensor(np.random.random((32, 2))))
    z, dlogp = flow(style, location)
    loss = torch.sum(dlogp)
    loss.backward()
