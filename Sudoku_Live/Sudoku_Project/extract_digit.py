import cv2 as cv
import imutils
import numpy as np
from skimage.segmentation import clear_border


def extract_digit(cell):
	# apply automatic thresholding to the cell and then clear any
	# connected borders that touch the border of the cell
    thresh = cv.threshold(cell, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    
    # find contours in the thresholded cell
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
	# if no contours were found than this is an empty cell
    if len(cnts) == 0:
        return None
	# otherwise, find the largest contour in the cell and create a
	# mask for the contour
    c = max(cnts, key=cv.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv.drawContours(mask, [c], -1, 255, -1)
    
    (h, w) = thresh.shape
    percentFilled = cv.countNonZero(mask) / float(w * h)
    if percentFilled < 0.03:
        return None

    digit = cv.bitwise_and(thresh, thresh, mask=mask)

    return digit