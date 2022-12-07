import os
import torch
from torch import optim, nn, utils, Tensor, device
import pytorch_lightning as pl
from IconFlow.iconflow.dataset import IconContourDataset
from torchinfo import summary
from src.colorizer_model import Colorizer


device = torch.device('cuda')
num_workers = 2 # Threads to use for data loading
# prefetch_factor = 2
dataset_dir = "./IconFlow/dataset"
batch_size = 32
image_size = 64
train_ratio = 0.9


if __name__ == "__main__":
    # Dataset initialization copied from iconflow __main__.py
    # https://github.com/djosix/IconFlow
    data_dir = os.path.join(dataset_dir, 'data')
    train_set = IconContourDataset(data_dir, image_size, split=(0, train_ratio))
    test_set = IconContourDataset(data_dir, image_size, split=(train_ratio, 1))
    net_train_set = IconContourDataset(
        data_dir,
        image_size,
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

    for image, contour in train_loader:
        print('image type:', type(image), 'image shape:', image.shape)
        print('contour type:', type(contour), 'contour shape:', contour.shape)
        break

    print("Num batches:", len(train_loader.dataset))

    # Define model
    # encoder = nn.Sequential(nn.Conv2d(1, 10, 3, stride=2, padding=1), nn.Conv2d(10, 20, 3, stride=2, padding=1))
    # decoder = nn.Sequential(nn.ConvTranspose2d(20, 10, 3, stride=2, padding=1, output_padding=1), nn.ConvTranspose2d(10, 3, 3, stride=2, padding=1, output_padding=1))
    # encoder = nn.Sequential(nn.Linear(image_size**2, 64), nn.ReLU(), nn.Linear(64, 16))
    # decoder = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 3*image_size**2))

    # summary(encoder, (1, image_size, image_size))
    # summary(decoder, (20, image_size//4, image_size//4))
    # model = TestModel(encoder, decoder)

    model = Colorizer()
    summary(model.contour_encoder, input_size=(batch_size, 1, image_size, image_size))

    trainer = pl.Trainer(max_epochs=10, accelerator="gpu")
    trainer.fit(model=model, train_dataloaders=train_loader)

