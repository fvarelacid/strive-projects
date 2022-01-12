import cv2 as cv
from helpers import *
from model import CNN
import torch


### Load the model
model = CNN()
model.load_state_dict(torch.load('output/best_model.pt'))
model.eval()

### Open Video Capture ###
cap = cv.VideoCapture("resources/videos/Sudoku_Vid5.MOV")

cv.namedWindow("Window")

### Check if camera is opened ###
if (cap.isOpened() == False):
  print("Error opening camera.")


prev_sudoku = None
### Read until video is completed ###
while True:
  ret, frame = cap.read()

  if ret == True:

    result_frame = sudoku_finder(frame, model)
 
    cv.imshow("Sudoku Result", result_frame)


    if cv.waitKey(1) & 0xFF == ord('q'):
      break
  
  else:
    break

cap.release()
cv.destroyAllWindows()