import cv2 as cv
import imutils
import numpy as np
from skimage.segmentation import clear_border
import math


def crop_img(img, ratio):
	height = img.shape[0]
	width = img.shape[1]

	height_ratio = math.floor(height * ratio)
	width_ratio = math.floor(width * ratio)

	crop_img = img[height_ratio:height - height_ratio, width_ratio:width - width_ratio]

	return crop_img

def extract_digit(cell):
	# apply automatic thresholding to the cell and then clear any
	# connected borders that touch the border of the cell
    thresh = cv.threshold(cell, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    
    # find contours in the thresholded cell
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

	
    if len(cnts) == 0:
        return None
	

    c = max(cnts, key=cv.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv.drawContours(mask, [c], -1, 255, -1)
    
    (h, w) = thresh.shape
    percentFilled = cv.countNonZero(mask) / float(w * h)
    if percentFilled < 0.03:
        return None

    digit = cv.bitwise_and(thresh, thresh   , mask=mask)
    digit = crop_img(digit, 0.11)

    _, digit = cv.threshold(digit, 200, 255, cv.THRESH_BINARY)
    digit = digit.astype(np.uint8)
    digit = cv.bitwise_not(digit)


    return digit