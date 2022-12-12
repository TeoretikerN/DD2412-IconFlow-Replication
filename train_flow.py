import multiprocessing
import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim, nn, utils, Tensor, device
from torchinfo import summary
from IconFlow.iconflow.dataset import IconContourDataset, StylePaletteDataset
from src.colorizer_model import Colorizer
from src.flow_model import NormalizingFlow


device = torch.device('cuda')
num_workers = multiprocessing.cpu_count() # Threads to use for data loading
dataset_dir = "./IconFlow/dataset"
batch_size = 32
image_size = 64
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
        sampler=utils.data.RandomSampler(flow_train_set, replacement=True, num_samples=len(train_set)),
        pin_memory=(device.type == 'cuda'),
        num_workers=num_workers
    )

    for image, location in train_loader:
        print('image type:', type(image), 'image shape:', image.shape)
        print('location type:', type(location), 'location shape:', location.shape)
        break

    colorizer = Colorizer()
    flow = NormalizingFlow(colorizer)
    summary(flow)

    logger = TensorBoardLogger("iconflow_logs", name="flow")
    trainer = pl.Trainer(logger=logger, max_epochs=200, accelerator="gpu")
    trainer.fit(model=flow, train_dataloaders=train_loader)
