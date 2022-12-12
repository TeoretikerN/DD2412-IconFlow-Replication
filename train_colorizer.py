import multiprocessing
import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim, nn, utils, Tensor, device
from torchinfo import summary
from IconFlow.iconflow.dataset import IconContourDataset
from src.colorizer_model import Colorizer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_workers = multiprocessing.cpu_count() # Threads to use for data loading
# prefetch_factor = 2
dataset_dir = "./IconFlow/dataset"
batch_size = 64
image_size = 64
validation_images = 10
train_ratio = 0.9

if __name__ == "__main__":
    # Dataset initialization copied from iconflow __main__.py
    # https://github.com/djosix/IconFlow
    data_dir = os.path.join(dataset_dir, 'data')
    train_set = IconContourDataset(data_dir, image_size, split=(0, train_ratio))
    test_set = IconContourDataset(data_dir, image_size, split=(train_ratio, 1))
    net_train_set = IconContourDataset(
        data_dir, image_size,
        random_crop=True,
        random_transpose=True,
        random_color=True,
        split=(0, train_ratio),
        normalize=True,
    )
    train_loader = utils.data.DataLoader(
        net_train_set,
        batch_size=batch_size,
        sampler=utils.data.RandomSampler(net_train_set),
        pin_memory=(device.type == 'cuda'),
        num_workers=num_workers
    )

    val_loader = utils.data.DataLoader(
        train_set,
        batch_size=validation_images,
        sampler=utils.data.RandomSampler(train_set, num_samples=validation_images),
        pin_memory=(device.type == 'cuda'),
        num_workers=num_workers
    )

    for image, contour in train_loader:
        print(type(image))
        print(type(contour))
        print(image.shape)
        print(contour.shape)
        break

    model = Colorizer()
    summary(model.contour_encoder, input_size=(batch_size,1,image_size,image_size))

    logger = TensorBoardLogger("iconflow_logs", name="colorizer")
    trainer = pl.Trainer(logger=logger, max_epochs=1000, accelerator="gpu")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
