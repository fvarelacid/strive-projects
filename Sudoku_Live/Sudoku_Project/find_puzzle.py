import imutils
import cv2 as cv
from imutils.perspective import four_point_transform
from helpers import *


def find_puzzle(image):
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (1, 1), 0)
    
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    thresh = cv.bitwise_not(thresh)


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

            approx = sorted(approx, key=lambda x: x[0][0])

            tl, bl = sorted(approx[:2], key=lambda x: x[0][1])
            tr, br = sorted(approx[2:], key=lambda x: x[0][1])

            tl = tuple(tl[0])
            tr = tuple(tr[0])
            bl = tuple(bl[0])
            br = tuple(br[0])

            eps = 1.08

            if side_lengths_are_too_different(tl, tr, br, bl, eps):
                continue

            else:
                break

    if puzzleCnt is None:
        raise Exception("Could not find Sudoku puzzle.")

    # apply a four point perspective transform to both the original
	# image and grayscale image to obtain a top-down bird's eye view
	# of the puzzle
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))

	# return a 2-tuple of puzzle in both RGB and grayscale
    return (puzzle, warped)