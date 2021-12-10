import cv2 as cv
from numpy.lib.function_base import extract
from extract_puzzle import extract_puzzle
from find_puzzle import find_puzzle
from extract_digit import extract_digit
from helpers import *
from sudoku import Sudoku
from tensorflow.keras.models import load_model

model = load_model('output/digit_classifier.h5')

### Open Video Capture ###
cap = cv.VideoCapture("resources/videos/Sudoku_Vid.MOV")

cv.namedWindow("Window")

### Check if camera is opened ###
if (cap.isOpened() == False):
  print("Error opening camera.")


### Read until video is completed ###
while True:
  ret, frame = cap.read()

  if ret == True:

    clone_frame = np.copy(frame)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_contour = None

    for contour in contours:
      area = cv.contourArea(contour)
      if area > 10000:
        if area > max_area:
          max_area = area
          max_contour = contour
    
    if max_contour is not None:
      # (puzzle, warped) = find_puzzle(frame)
      (board, cellLocs) = extract_puzzle(frame, model)
      # cv.imshow("Puzzle", warped)
      # find_corners_of_contour(max_contour)
      print(board)
      
    cv.imshow("Window", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
      break
  
  else:
    break

cap.release()
cv.destroyAllWindows()