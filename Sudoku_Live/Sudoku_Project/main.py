import cv2 as cv

### Open Video Capture ###
cap = cv.VideoCapture(0)

### Check if camera is opened ###
if (cap.isOpened()== False):
  print("Error opening camera.")

### Read until video is completed ###
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # Converting the input frame to grayscale
    gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Fliping the image as said in question
    gray_flip = cv.flip(gray,1)

    # Display the resulting frame
    cv.imshow('Frame',gray_flip)

    # Press Q on keyboard to exit
    if cv.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv.destroyAllWindows()
