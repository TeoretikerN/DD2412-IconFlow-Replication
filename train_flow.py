import os
import torch
from torch import optim, nn, utils, Tensor, device
import pytorch_lightning as pl
from IconFlow.iconflow.dataset import IconContourDataset, StylePaletteDataset
from IconFlow.iconflow.utils.train import random_sampler
from torchinfo import summary
from src.colorizer_model import Colorizer
from src.flow_model import NormalizingFlow


device = torch.device('cuda')
num_workers = 4 # Threads to use for data loading
dataset_dir = "./IconFlow/dataset"
batch_size = 32
image_size = 128
train_ratio = 0.9
max_samples = 1000


if __name__ == "__main__":
    # Dataset initialization copied from iconflow __main__.py
    # https://github.com/djosix/IconFlow
    data_dir = os.path.join(dataset_dir, 'data')
    train_set = IconContourDataset(data_dir, image_size, split=(0, train_ratio))
    test_set = IconContourDataset(data_dir, image_size, split=(train_ratio, 1))
    flow_train_set = StylePaletteDataset(
        data_dir,
        image_size,
        dataset_dir,
        max_samples,
        num_workers=num_workers
    )
    train_loader = utils.data.DataLoader(
        flow_train_set,
        batch_size=batch_size,
        sampler=random_sampler(len(flow_train_set)),
        pin_memory=(device.type == 'cuda'),
        num_workers=num_workers
    )

    for image, contour in train_loader:
        print(type(image))
        print(type(contour))
        print(image.shape)
        print(contour.shape)
        break

    colorizer = Colorizer()
    summary(colorizer.contour_encoder, input_size=(batch_size, 1, image_size, image_size))

    flow = NormalizingFlow(colorizer)
    summary(flow.flow)

    trainer = pl.Trainer(max_epochs=1, accelerator="gpu")
    trainer.fit(model=flow, train_dataloaders=train_loader)
