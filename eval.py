# import pytorch_lightning as pl
# from IconFlow.iconflow.utils.dataset import _get_contour as get_contour
from src.colorizer_model import Colorizer
from torchinfo import summary
from src.structure_distance import structure_distance, get_contour
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.feature import canny
from IconFlow.iconflow.dataset import IconContourDataset
import os
from torch import utils, device
import torch
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance


dataset_dir = "./IconFlow/dataset"
checkpoint = "./colorizer_resnet18_epoch=999-step=179000.ckpt"

def compute_fid(original_batches, out_batches):
	print("Computing FID...")
	fid = FrechetInceptionDistance(feature=192)
	for i in tqdm(range(len(out_batches))):
		img_batch = original_batches[i]
		out_batch = out_batches[i]
		fid.update(torch.tensor(img_batch, dtype=torch.uint8), real=True)
		fid.update(torch.tensor(out_batch, dtype=torch.uint8), real=False)
	return fid.compute()

def compute_metrics(original_contiguous, out_contiguous):
	print("Computing other metrics...")
	s_dists = []
	for i in tqdm(range(len(out_contiguous))):
		img = original_contiguous[i]
		out = out_contiguous[i]
		s_dists.append(structure_distance(img, out))
	return np.mean(s_dists)

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

def show_test_images(original_contiguous, out_contiguous):
	for i in range(5):
		img = original_contiguous[i]
		out = out_contiguous[i]
		plt.subplot(4, 5, i+1), plt.imshow(img)
		plt.title('Original '+str(i)), plt.xticks([]), plt.yticks([])
		plt.subplot(4, 5, 5+i+1), plt.imshow(get_contour(img))
		plt.title('Original cont '+str(i)), plt.xticks([]), plt.yticks([])
		plt.subplot(4, 5, 10+i+1), plt.imshow(out)
		plt.title('Reconstruction '+str(i)), plt.xticks([]), plt.yticks([])
		plt.subplot(4, 5, 15+i+1), plt.imshow(get_contour(out))
		plt.title(f'Dist: {int(structure_distance(img, out))}'), plt.xticks([]), plt.yticks([])
	plt.show()


def to_8bit_rgb(images):
	# Takes images in range -0.5 to 0.5 and converts to 0-255
	images = (images+0.5)*255
	return np.clip(images, 0, 255).astype(np.uint8)


if __name__ == "__main__":
	train_ratio = 0.90

	batch_size = 128
	image_size = 64

	model = Colorizer.load_from_checkpoint(checkpoint)
	# summary(model)
	model.eval()
	out_batches = []
	original_batches = []
	print("Generating reconstructions...")
	test_loader = get_test_loader(batch_size, image_size, train_ratio)
	for batch in tqdm(test_loader):
		images, contours = batch
		with torch.no_grad():
			_out, out_cc = model(contours, images)
		# Convert to numpy
		out_cc = np.array(out_cc)
		images = np.array(images)
		# Get in uint8, 0-255 rgb
		out_batches.append(to_8bit_rgb(out_cc))
		original_batches.append(to_8bit_rgb(images))

	# For other metrics than FID, get in order N W H C
	original_contiguous = np.concatenate(original_batches)
	out_contiguous = np.concatenate(out_batches)
	original_contiguous = np.moveaxis(original_contiguous, 1, -1)
	out_contiguous = np.moveaxis(out_contiguous, 1, -1)

	print("Frechet inception distance:", compute_fid(original_batches, out_batches))
	sdist = compute_metrics(original_contiguous, out_contiguous)
	print("Mean structure distance:", sdist)

	