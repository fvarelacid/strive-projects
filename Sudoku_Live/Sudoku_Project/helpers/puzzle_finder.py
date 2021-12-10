from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import skimage
import numpy as np
import imutils
import cv2 as cv
from preprocessing import preprocessImage, centeringImage


def find_puzzle(image, debug=False):
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), 0)
    
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    thresh = cv.bitwise_not(thresh)

    # if debug:
    #     cv.imshow("Puzzle Thresh", thresh)
    #     cv.waitKey(0)

    # finding countours in the image and sort by descending size
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)

    puzzleCnt = None

    for c in cnts:
        # approximate the contour
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our puzzle
        if len(approx) == 4:
            puzzleCnt = approx
            break

    if puzzleCnt is None:
        raise Exception("Could not find Sudoku puzzle.")

    	
    # apply a four point perspective transform to both the original
	# image and grayscale image to obtain a top-down bird's eye view
	# of the puzzle
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))
	# check to see if we are visualizing the perspective transform
    # if debug:
	# 	# show the output warped image (again, for debugging purposes)
    #     cv.imshow("Puzzle Transform", warped)
    #     cv.waitKey(0)
	# return a 2-tuple of puzzle in both RGB and grayscale
    return (puzzle, warped)



def extract_digit(cell, debug=False):
	# apply automatic thresholding to the cell and then clear any
	# connected borders that touch the border of the cell
    thresh = cv.threshold(cell, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
	# check to see if we are visualizing the cell thresholding step
    if debug:
        cv.imshow("Cell Thresh", thresh)
        cv.waitKey(0)
    
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

    if debug:
        # show the digit extracted from the cell
        cv.imshow("Digit", digit)
        cv.waitKey(0)

    return digit