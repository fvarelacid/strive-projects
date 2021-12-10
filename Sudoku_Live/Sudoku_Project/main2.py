from sudoku import Sudoku
import numpy as np
import argparse
import imutils
import cv2
from torch.nn.modules.activation import Softmax
from helpers.puzzle_finder import extract_digit
from helpers.puzzle_finder import find_puzzle
import torch.functional as F
import torch.nn as nn
import torch
from preprocessing import *
from helpers.puzzle_finder import *
from collections import OrderedDict


# class Model:
#     def build(input_size, hidden_sizes, output_size):

#         model = nn.Sequential(OrderedDict([
#             ('fc1', nn.Linear(input_size, hidden_sizes[0])),
#             ('relu1', nn.ReLU()),
#             ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
#             ('relu2', nn.ReLU()),
#             ('logits', nn.Linear(hidden_sizes[1], output_size))]))
#         return model

# Hyperparameters for our network
input_size   = 784
hidden_sizes = [128, 64]
output_size  = 10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU())  # on device cpu
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU())  # on device "cuda:1"
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_sizes[1], output_size))  # on device "cuda:0"
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)


model = torch.load('output/model.pt')
model.eval()

# # Start capturing the video
# cap = cv2.VideoCapture(0)

# if __name__ == "__main__":
#     detected = False
#     solved = False
    
img = cv.imread("images/Sudoku4.png")
image = imutils.resize(img, width=600)

(puzzleImage, warped) = find_puzzle(image, debug=False)

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
board_list = []

# loop over the grid locations
for y in range(0, 9):
	# initialize the current list of cell locations
    row = []
    row_board = []
    for x in range(0, 9):
		# compute the starting and ending (x, y)-coordinates of the
		# current cell
        startX = x * stepX
        startY = y * stepY
        endX = (x + 1) * stepX
        endY = (y + 1) * stepY
		# add the (x, y)-coordinates to our cell locations list
        row.append((startX, startY, endX, endY))

        # crop the cell from the warped transform image and then
		# extract the digit from the cell
        cell = warped[startY:endY, startX:endX]
        digit = extract_digit(cell, debug=False)
		# verify that the digit is not empty
        if digit is not None:
			# resize the cell to 28x28 pixels and then prepare the
			# cell for classification
            resized = cv2.resize(digit, (28, 28))
            resized = resized.resize_(resized.size()[0], 784)

            with torch.no_grad:
                logits = model.forward(resized)
            
            ps = F.softmax(logits, dim=1)
            print(ps.argmax().item())

        else:
            row_board.append(0)

	# add the row to our cell locations
    cellLocs.append(row)
    board_list.append(row_board)

print(board_list)