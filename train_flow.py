import multiprocessing
import numpy as np
import os
import pytorch_lightning as pl
import random
import torch
import torchvision.transforms.functional as F
from glob import glob
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim, nn, utils, Tensor, device
from torchinfo import summary
from IconFlow.iconflow.dataset import IconContourDataset, StylePaletteDataset
from IconFlow.iconflow.utils.style import get_style_image
from src.colorizer_model import Colorizer
from src.flow_model import NormalizingFlow


device = torch.device('cuda')
num_workers = multiprocessing.cpu_count() # Threads to use for data loading
dataset_dir = "./IconFlow/dataset"
batch_size = 64
image_size = 64
train_ratio = 0.9
max_samples = 1000
colorizer_version = 12

if __name__ == "__main__":
    # Get the path for the colorizer model checkpoint
    colorizer_path = f'iconflow_logs/colorizer/version_{colorizer_version}/checkpoints/'
    colorizer_path = sorted(glob(colorizer_path + '*.ckpt'))[-1]

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


    class FlowValidationSet(utils.data.IterableDataset):
        def __init__(self, validation_images=4):
            self.validation_images = validation_images

        def from_image(self, image):
            return F.to_tensor(image) - 0.5

        def __iter__(self):
            val_images = self.validation_images
            condition_train = random.choice(train_set)[1]
            condition_test = random.choice(test_set)[1]
            condition_batch = [condition_train] * val_images + [condition_test] * val_images
            condition = torch.stack(condition_batch)

            selected_style_names = np.random.choice(flow_train_set.style_names, size=(val_images * 2))

            selected_style_images = torch.stack([
                self.from_image(get_style_image(flow_train_set.style_to_cmb[name], name))
                for name in selected_style_names
            ])

            location = torch.stack([
                flow_train_set.position_to_condition(flow_train_set.style_to_pos[name])
                for name in selected_style_names
            ])

            yield condition, selected_style_images, location

    val_loader = utils.data.DataLoader(
        FlowValidationSet(),
        #batch_size=validation_images,
        #sampler=utils.data.RandomSampler(train_set, num_samples=validation_images),
        pin_memory=(device.type == 'cuda'),
        num_workers=num_workers
    )

    for image, location in train_loader:
        print('image type:', type(image), 'image shape:', image.shape)
        print('location type:', type(location), 'location shape:', location.shape)
        break

    # Load the trained colorizer
    colorizer = Colorizer.load_from_checkpoint(colorizer_path)

    flow = NormalizingFlow(colorizer)
    summary(flow)

    logger = TensorBoardLogger("iconflow_logs", name="flow")
    trainer = pl.Trainer(logger=logger, max_epochs=1000, accelerator="gpu")
    trainer.fit(model=flow, train_dataloaders=train_loader, val_dataloaders=val_loader)
