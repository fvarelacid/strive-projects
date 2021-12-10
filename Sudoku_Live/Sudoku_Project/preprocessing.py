import torchvision.transforms as T
import cv2 as cv
import numpy as np

def convert_img_tensor(img):

    convert_tensor = T.Compose([
    T.Resize(size=(28, 28)),
    T.Grayscale(num_output_channels=1),
    T.ToTensor(),
    T.Normalize((0.5), (0.5))
    ])

    return convert_tensor(img)


def preprocessImage(image, skip_dilation = False):
	preprocess = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	preprocess = cv.adaptiveThreshold(preprocess, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, 2)

	if not skip_dilation:
		kernel = np.array([[0., 1., 0.], [1., 2., 1.], [0., 1., 0.]], dtype = np.uint8)
		preprocess = cv.dilate(preprocess, kernel)
	return preprocess


def centeringImage(image):
	rows = image.shape[0]
	for i in range(rows):
		#Floodfilling the outermost layer
		cv.floodFill(image, None, (0, i), 0)
		cv.floodFill(image, None, (i, 0), 0)
		cv.floodFill(image, None, (rows-1, i), 0)
		cv.floodFill(image, None, (i, rows-1), 0)
		#Floodfilling the penultimate layer
		cv.floodFill(image, None, (1, i), 0)
		cv.floodFill(image, None, (i, 1), 0)
		cv.floodFill(image, None, (rows-2, i), 0)
		cv.floodFill(image, None, (i, rows-2), 0)


	top = None
	bottom = None
	left = None
	right = None
	threshold = 50
	center = rows // 2
	
	for i in range(center, rows):
		if bottom is None:
			temp = image[i]
			if sum(temp) < threshold or i == rows - 1:
				bottom = i
		if top is None:
			temp = image[rows - i - 1]
			if sum(temp) < threshold or i == rows - 1:
				top = rows - i - 1
		if left is None:
			temp = image[:, rows - i - 1]
			if sum(temp) < threshold or i == rows - 1:
				left = rows - i - 1
		if right is None:
			temp = image[:, i]
			if sum(temp) < threshold or i == rows - 1:
				right = i
	if (top == left and bottom == right):
		return 0, image

	image = image[top - 5:bottom + 5, left - 5:right + 5]
	return 1, image    
