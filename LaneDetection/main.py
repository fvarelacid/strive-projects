import numpy as np
import matplotlib.pyplot as plt
import cv2

def imshow(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(15,15))
    plt.imshow(img)


def grayscale(img):
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
def canny(img):
    img_blur = cv2.GaussianBlur(img, (5,5), cv2.BORDER_DEFAULT)
    canny = cv2.Canny(img_blur,100,255)
    return canny


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), cv2.BORDER_DEFAULT)

def region_of_interest(img, vertices):
    mask_zeros = np.zeros_like(img)
    cv2.fillPoly(mask_zeros, np.int32([vertices]), 255)
    img_masked = cv2.bitwise_and(img, mask_zeros)
    return img_masked


def draw_lines(img, lines, color=[0, 0, 255], thickness=6):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    return line_img


def hough_lines(img, rho, theta, threshold, lines_array, min_line_len, max_line_gap):
    return cv2.HoughLinesP(img, rho, theta, threshold, lines_array, min_line_len, max_line_gap)


def weighted_img(img, line_img, α=0.8, β=1., γ=1.):
    return cv2.addWeighted(img, α, line_img, β, γ)

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = ((y1 - intercept)/slope)
    x2 = ((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for x1, y1, x2, y2 in lines:
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


###### FOR IMAGE TESTING #######

test_image = cv2.imread('test_images/solidWhiteCurve.jpg')
def main(image_img):
    lane_img = np.copy(image_img)

    height = lane_img.shape[0]
    width = lane_img.shape[1]

    vertices = np.array([
        [150, height],
        [500, 290],
        [width,height]]
    )

    lane_img_gray = grayscale(lane_img)

    can = canny(lane_img_gray)
    masked_img = region_of_interest(can, vertices)
    lines = hough_lines(masked_img, 2, np.pi/180, 100, np.array([]), min_line_len=40, max_line_gap=53)
    line_img = draw_lines(lane_img, lines)
    final_img = weighted_img(lane_img, line_img)
    return (final_img)



###### FOR VIDEO TESTING #######

cap = cv2.VideoCapture('test_videos/solidYellowLeft.mp4')

while(cap.isOpened()):

    ret, frame = cap.read() 
    #cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    if ret:
        cv2.imshow('image',main(frame))
    else:
       print('no video')
       cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()