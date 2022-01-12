import cv2
import numpy as np
import math
import cv2 as cv
from extract_digit import extract_digit
from imutils.perspective import four_point_transform
from torchvision import transforms
import copy
import sudoku_solver

def prepare_for_pred(img_array):
	digit = cv.resize(img_array, (28, 28))
	transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])
	digit = transform(digit)
	digit = digit.unsqueeze(1)
	return digit


def find_corners_of_contour(contours, corner_amount=4, max_iter=200):

    coefficient = 1
    while max_iter > 0 and coefficient >= 0:
        max_iter = max_iter - 1

        epsilon = coefficient * cv2.arcLength(contours, True)

        poly_approx = cv2.approxPolyDP(contours, epsilon, True)
        hull = cv2.convexHull(poly_approx)
        if len(hull) == corner_amount:
            return hull
        else:
            if len(hull) > corner_amount:
                coefficient += .01
            else:
                coefficient -= .01
    return None


def side_lengths_are_too_different(A, B, C, D, tolScale):
    AB = math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
    AD = math.sqrt((A[0]-D[0])**2 + (A[1]-D[1])**2)
    BC = math.sqrt((B[0]-C[0])**2 + (B[1]-C[1])**2)
    CD = math.sqrt((C[0]-D[0])**2 + (C[1]-D[1])**2)
    shortest = min(AB, AD, BC, CD)
    longest = max(AB, AD, BC, CD)
    return longest > tolScale * shortest

def approx_90_degrees(angle, epsilon):
    return abs(angle - 90) < epsilon

def angle_between(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector2 = vector_2 / np.linalg.norm(vector_2)
    dot_droduct = np.dot(unit_vector_1, unit_vector2)
    angle = np.arccos(dot_droduct)
    return angle * 57.2958  # Convert to degree

def find_coordinates(corners):
	approx = sorted(corners, key=lambda x: x[0][0])
	
	tl, bl = sorted(approx[:2], key=lambda x: x[0][1])
	tr, br = sorted(approx[2:], key=lambda x: x[0][1])
	
	tl = tuple(tl[0])
	tr = tuple(tr[0])
	bl = tuple(bl[0])
	br = tuple(br[0])

	return np.array([tl, tr, br, bl], dtype = np.float32)


def sudoku_finder(frame, model):

	new_frame = frame.copy()
	
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	blur = cv.GaussianBlur(gray, (5, 5), 0)
	thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
	
	contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	
	max_area = 0
	max_contour = None
	
	### Check for the biggest contour with area bigger than 10000
	for contour in contours:
		area = cv.contourArea(contour)
		if area > 10000:
			if area > max_area:
				max_area = area
				max_contour = contour
				
	if max_contour is None:
		return frame

	### Find the corners of the biggest contour
	corners = find_corners_of_contour(max_contour, 4)

	if corners is None:
		return frame

	### Get corners coordinates
	corners_coord = find_coordinates(corners)

	tl = corners_coord[0]
	tr = corners_coord[1]
	br = corners_coord[2]
	bl = corners_coord[3]


	### Check for corners angles
	tolAngle = 20
	if not (approx_90_degrees(angle_between(tr - tl, bl - tl), tolAngle) and approx_90_degrees(angle_between(tl - tr, br - tr), tolAngle) and approx_90_degrees(angle_between(tr - br, bl - br), tolAngle) and approx_90_degrees(angle_between(tl - bl, br - bl), tolAngle)):
		return frame

	### Check for sides
	if side_lengths_are_too_different(tl, tr, br, bl, tolScale=1.1):
		return frame	
	
	### Apply bird view
	warped = four_point_transform(blur, corners.reshape(4, 2))
	warped = cv.adaptiveThreshold(warped, 255, 1, 1, 11, 2)
	warped = cv.bitwise_not(warped)
	_, warped = cv.threshold(warped, 150, 255, cv.THRESH_BINARY)

	board = np.zeros((9, 9), dtype="int")

	height = warped.shape[0] // 9
	width = warped.shape[1] // 9


	cellLocs = []

	### Run through the cells and predict numbers
	for y in range(0, 9):
		row = []
		for x in range(0, 9):

			startY = y * height
			startX = x * width
			endY = (y + 1) * height
			endX = (x + 1) * width

			row.append((tl[0] + startX, tl[1] + startY, tl[0] + endX, tl[1] + endY))

			cell = warped[startY:endY, startX:endX]
			
			digit = extract_digit(cell)

			if digit is not None:

				digit = prepare_for_pred(digit)

				pred = model(digit)[0].argmax(axis=1) + 1
				board[y, x] = pred

		cellLocs.append(row)

	board_copy = copy.deepcopy(board)

	### Solve the Sudoku board
	sudoku_solver.solve_sudoku(board)

	### Draw the Sudoku board on the image
	if(sudoku_solver.all_board_non_zero(board)):
		for (cellRow, boardRow, boardCRow) in zip(cellLocs, board, board_copy):
			for (box, digit, digit_copy) in zip(cellRow, boardRow, boardCRow):
				
				if(digit_copy != 0):
					continue
				
				startX, startY, endX, endY = box
				textX = int((endX - startX) * 0.2)
				textY = int((endY - startY) * -0.2)
				textX += startX
				textY += endY
				
				cv.putText(new_frame, str(digit), (int(textX), int(textY)), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)

	return new_frame