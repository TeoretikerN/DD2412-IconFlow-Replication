import numpy as np
from skimage.feature import canny
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from PIL import Image

def get_contour(img):
	x = np.array(img)
	contour = 0
	for i, layer in enumerate(np.rollaxis(x, -1)):
		contour |= canny(layer, 0)
	return contour

def structure_distance(original_image, generated_image):
	original_edges = get_contour(original_image)
	generated_edges = get_contour(generated_image)
	original_edge_points = np.argwhere(original_edges)
	generated_edge_points = np.argwhere(generated_edges)
	distances = cdist(generated_edge_points, original_edge_points)
	total_min_distance = distances.min(axis=1).sum() + distances.min(axis=0).sum()
	return total_min_distance

if __name__ == "__main__":
	# Do some tests (these may no longer work)
	img_grayscale = np.array(Image.open('./IconFlow/dataset/data/64/img/000000.png').convert('L'))
	contour = np.array(Image.open('./IconFlow/dataset/data/64/contour/000000.png'))
	edges = canny(img_grayscale,1)
	print("Structure distance between edge image and true contour:")
	print(structure_distance(contour, img_grayscale))
	plt.subplot(131),plt.imshow(img_grayscale,cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(132),plt.imshow(edges,cmap = 'gray')
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(133),plt.imshow(contour>255/3,cmap = 'gray')
	plt.title('True Contour'), plt.xticks([]), plt.yticks([])
	plt.show()