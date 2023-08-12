import cv2
import numpy as np
import matplotlib.pyplot as plt


# Get the coordinates of lines
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


# Get single lines from the multiple lines
def slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        # Based on the test image , the slope will be >0 for the right fit
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


# Perform Canny Transformation on the original images
def cannyTransfrom(image):
    # Convert the image from BGR to RGB and greyscale
    im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, [5, 5], 0)
    # Blur for better results
    canny = cv2.Canny(img_blur, 50, 150)
    return canny

# Draw the red lines on the line detected
def displayResult(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            xl, yl, x2, y2 = line.reshape(4)
            cv2.line(line_image, (xl, yl), (x2, y2), (0, 0, 255), 8)
    return line_image

# Get only the region of Interest
def regionOfInterest(image):
    height = image.shape[0]
    width = image.shape[1]
    mask = np.zeros_like(image)
    # Values are optimized based on image using plt.imshow
    triangle = np.array([[(0, height), (513, 215), (1000, height)]], np.int32)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Test
image = cv2.imread('road2.jpg')
lane_image = np.copy(image)
canny = cannyTransfrom(lane_image)
cropped_image = regionOfInterest(canny)
# Probabilistic Hough Line Transform
lines = cv2.HoughLinesP(cropped_image, 1, np.pi / 180, 30, maxLineGap=200) # maximum gap between the lines
averaged_lines = slope_intercept(lane_image, lines)
line_image = displayResult(lane_image, averaged_lines)
combo_image = cv2.addWeighted(lane_image, 0.4, line_image, 1, 1)
cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.imshow("Area of Interest", cropped_image)
cv2.waitKey(0)
cv2.imshow("Result Image", combo_image)
cv2.waitKey(0)
