import pytesseract
import cv2 as cv
import pytesseract


image = cv.imread("resources/images/Sudoku3.png")
rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6 outputbase digits')
print(text)
