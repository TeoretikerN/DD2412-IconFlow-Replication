import numpy as np
import os
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import utils, device

from src.colorizer_model import Colorizer
from IconFlow.iconflow.dataset import IconContourDataset


colorizer_version = 2
dataset_dir = "./IconFlow/dataset"
checkpoint_dir = f'iconflow_logs/colorizer/version_{colorizer_version}/checkpoints/'
checkpoint = "epoch=999-step=179000.ckpt"


def get_test_loader(batch_size, image_size, train_ratio):
	data_dir = os.path.join(dataset_dir, 'data')
	test_set = IconContourDataset(data_dir, image_size, split=(train_ratio, 1))
	test_loader = utils.data.DataLoader(
	    test_set,
	    batch_size=batch_size,
	    # sampler=utils.data.RandomSampler(test_set),
	    pin_memory=(device.type == 'cuda')
	)
	return test_loader

def show_test_images(original_contiguous, out_contiguous, contours, batch_size, filename=''):
	for i in range(batch_size):
		img = original_contiguous[i]
		out = out_contiguous[i]
		contour = contours[i]
		plt.subplot(3, batch_size, i+1), plt.imshow(img)
		plt.title('Style'), plt.xticks([]), plt.yticks([])
		plt.subplot(3, batch_size, 5+i+1), plt.imshow(contour, cmap='gray')
		plt.title('Sketch'), plt.xticks([]), plt.yticks([])
		plt.subplot(3, batch_size, 10+i+1), plt.imshow(out)
		plt.title('Output'), plt.xticks([]), plt.yticks([])
	plt.show()
	if filename:  
		plt.savefig(filename)

def to_8bit_rgb(images):
	# Takes images in range -0.5 to 0.5 and converts to 0-255
	images = (images+0.5)*255
	return np.clip(images, 0, 255).astype(np.uint8)

def reorder_axes(images):
	images = np.swapaxes(images, 1, 3)
	return np.swapaxes(images, 1, 2)

if __name__ == "__main__":
	train_ratio = 0.90

	batch_size = 5
	image_size = 64

	model = Colorizer.load_from_checkpoint(checkpoint_dir+checkpoint)
	model.eval()
	test_loader = get_test_loader(batch_size*2, image_size, train_ratio)
	images, contours = next(iter(test_loader))
	images = images[0:batch_size]
	contours = contours[batch_size:]
	with torch.no_grad():
		out, _out_cc = model(contours, images)
        
    # Convert to numpy
	out = np.array(out)
	images = np.array(images)
	contours = np.array(contours)
    # Get in uint8, 0-255 rgb, and reorder axes for imshow
	out = reorder_axes(to_8bit_rgb(out))
	images = reorder_axes(to_8bit_rgb(images))
	contours = reorder_axes(contours)
    
	show_test_images(images, out, contours, batch_size, filename='reconstruction_test.png')