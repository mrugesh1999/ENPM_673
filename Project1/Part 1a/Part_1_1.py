# Importing required libraries
import cv2
from matplotlib import pyplot as plt
import numpy as np


# Creating function to scale the magnitude spectrum values to uint8 without loss of data
def convert(img, target_type_min, target_type_max, target_type):

    """
    :param img: Source image
    :param target_type_min: 0 for uint8
    :param target_type_max: 255 for uint8
    :param target_type: uint8 as cv2 supports that format
    :return: image with given arguments
    """
    imin = img.min()
    imax = img.max()
    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


# Sourcing the video
Video = cv2.VideoCapture('Tag1.mp4')

# Getting a single frame to perform operation (frame npo. 50)
Video.set(1, 50)
is_success, frame = Video.read()

# Converting that frame to grayscale to perform FFT
frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
ret, frame_ = cv2.threshold(frame_, 240, 255, cv2.THRESH_BINARY)


# FFT is a NP array with 2 Dimensions (Rows and columns)
# First channel consists of real part and second channel is imaginary part of FFT
# Important to note that FFT is an algorithm to find DFT
FFT = cv2.dft(np.float32(frame_), flags=cv2.DFT_COMPLEX_OUTPUT)

# Shifting array so that the center represents the zero coordinate
FFT_shift_ = np.fft.fftshift(FFT)
mag_spec_ = 20 * np.log(cv2.magnitude(FFT_shift_[:, :, 0], FFT_shift_[:, :, 1]) + 0.000001)
# Creating mask
# The circular mask with center at (0, 0)
# The radius determine the cutoff frequency

# Defining no. od rows and columns
r, c = frame_.shape

# Finding center of frame
CenR, CenC = int(r/2), int(c/2)

# Note that the mask is a 2D array with 2 channels
mask = np.ones((r, c, 2), np.uint8)

# The radii is the parameter to change cutoff
radii = 80

# This function will generate a 2D grid of defined rows and columns
X, Y = np.ogrid[:r, :c]

# Creating a masking circle by assigning values 0 that fall in the range defined
masking_portion = (X - CenR) ** 2 + (Y - CenC) ** 2 <= radii*radii
mask[masking_portion] = 0

# apply mask to the FFT
FFT_shift = FFT_shift_ * mask

mag_spec = 20 * np.log(cv2.magnitude(FFT_shift[:, :, 0], FFT_shift[:, :, 1]) + 0.0000001)

# Applying inverse FFT
FFT_is_shift = np.fft.ifftshift(FFT_shift)

# Getting the modified frame back
frame_back_ = cv2.idft(FFT_is_shift)
frame_back = cv2.magnitude(frame_back_[:, :, 0], frame_back_[:, :, 1])
frame_back = np.array(frame_back)

# Using function defined to get supported image
frame_back = convert(frame_back, 0, 255, 'uint8')


# Manually thresholding image to binary*
for i in range(frame_back.shape[0]):
    for j in range(frame_back.shape[1]):
        if frame_back[i, j] >= 110:
            frame_back[i, j] = 255
        else:
            frame_back[i, j] = 0

# Finding contours in that image
contours, hierarchy = cv2.findContours(frame_back, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Selecting the right contour
contours = contours[3]

# Finding extreme points (corners)
i_min = 100000
i_max = 2
j_min = 20000000000000
j_max = 2

for i in range(contours.shape[0]):
    val = contours[i, 0, 0]
    if val >= i_max:
        i_max = contours[i, 0, 0]
        j_i_max = contours[i, 0, 1]
    if val <= i_min:
        i_min = contours[i, 0, 0]
        j_i_min = contours[i, 0, 1]


for i in range(contours.shape[0]):
    val = contours[i, 0, 1]
    if val >= j_max:
        j_max = contours[i, 0, 1]
        i_j_max = contours[i, 0, 0]
    if val <= j_min:
        j_min = contours[i, 0, 1]
        i_j_min = contours[i, 0, 0]

# Cropping the images
frame_back_new = frame_back[j_min:j_max, i_min:i_max]
frame_crop = frame[j_min:j_max, i_min:i_max]

# Plotting them in single window
# Figure size is set to open 30% window in 4k resolution
fig = plt.figure(figsize=(7, 7))

# The input frame is plotted
# This is a 3*2 grid and placed at 1st spot (3, 2, 1)
IN = fig.add_subplot(3, 2, 1)
IN.imshow(frame, cmap='gray')
IN.title.set_text('Input Image')

OUT1 = fig.add_subplot(3, 2, 2)
OUT1.imshow(mag_spec_, cmap='gray')
OUT1.title.set_text('FFT of image (Centralized)')

OUT2 = fig.add_subplot(3, 2, 3)
OUT2.imshow(mag_spec, cmap='gray')
OUT2.title.set_text('Applied Mask')

OUT3 = fig.add_subplot(3, 2, 4)
OUT3.imshow(frame_back, cmap='gray')
OUT3.title.set_text('Inverse FFT with mask ON')

OUT4 = fig.add_subplot(3, 2, 5)
OUT4.imshow(frame_back_new, cmap='gray')
OUT4.title.set_text('Cropped AR tag (Inverse FFT)')

OUT5 = fig.add_subplot(3, 2, 6)
OUT5.imshow(frame_crop)
OUT5.title.set_text('Cropped AR tag (Original Frame)')

# Showing the final figure
plt.show()
