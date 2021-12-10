# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2 as cv
from extract_digit import extract_digit
from find_puzzle import find_puzzle


def extract_puzzle(frame, model):
    (puzzleImage, warped) = find_puzzle(frame)
    # initialize our 9x9 Sudoku board
    board = np.zeros((9, 9), dtype="int")
    # a Sudoku puzzle is a 9x9 grid (81 individual cells), so we can
    # infer the location of each cell by dividing the warped image
    # into a 9x9 grid
    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9
    # initialize a list to store the (x, y)-coordinates of each cell
    # location
    cellLocs = []

    # loop over the image in horizontal strips
    for y in range(0,9):
        row = []

        # loop over the image in vertical strips
        for x in range(0,9):

            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY

            row.append((startX, startY, endX, endY))

            cell = warped[startY:endY, startX:endX]
            digit = extract_digit(cell)

            if digit is not None:
                
                digit = cv.resize(digit, (28, 28))
                digit = digit.astype("float") / 255.0
                digit = img_to_array(digit)
                digit = np.expand_dims(digit, axis=0)

                pred = model.predict(digit).argmax(axis=1)[0]
                board[y, x] = pred
        
        cellLocs.append(row)
    
    # return the Sudoku puzzle
    return (board, cellLocs)