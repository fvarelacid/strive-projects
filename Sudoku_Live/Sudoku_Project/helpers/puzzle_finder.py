from skimage import data, io, filters, measure, morphology, segmentation
import numpy as np
import imutils
import cv2


def find_puzzle(image):
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    return edged

    # # find the contours in the edged image, keeping only the
    # # largest ones, and initialize the screen contour
    # cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # # loop over the contours
    # for c in cnts:
    #     # approximate the contour
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    #     # if our approximated contour has four points, then we
    #     # can assume that we have found our screen
    #     if len(approx) == 4:
    #         screenCnt = approx
    #         break

    # # apply the four point transform to obtain a top-down
    # # view of the original image
    # warped = four_point_transform(image, screenCnt.reshape(4, 2))

    # # convert the warped image to grayscale, then threshold it
    # # to give it that 'black and white' paper effect
    # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # warped = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # # make a copy of the original image
    # output = image.copy()

    # # find contours in the thresholded image, then initialize
    # # the list of


find_puzzle('images/Sudoku1.HEIC')