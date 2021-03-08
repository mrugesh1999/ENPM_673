# Importing required libraries
import cv2
import numpy as np

# Reading the image in color and grayscale
image = cv2.imread('/home/zeus/Desktop/ref_marker.png')
img = cv2.imread('/home/zeus/Desktop/ref_marker.png', 0)

# Getting the shape of each element in grid from image
x = img.shape[0] / 8
y = img.shape[1] / 8

# Creating an 8x8 grid
grid_data = np.zeros((8, 8))
pixel_list = []

# Getting the data of middle pixel in each pixel group of image wrt grid
for i in range(8):
    # y axis reset
    y_axis = 12.5
    for j in range(8):
        grid_data[i, j] = img[int(12.5 + (25*i)), int(12.5 + (25*j))]


# Removing padding
grid_without_padding = grid_data[2:6, 2:6]

# Getting the orientation of the tag
if grid_without_padding[0, 0] == 255:
    orientation = 'TL'
    print('TL')
if grid_without_padding[0, 3] == 255:
    orientation = 'TR'
    print('TR')
if grid_without_padding[3, 0] == 255:
    orientation = 'BL'
    print('BL')
if grid_without_padding[3, 3] == 255:
    orientation = 'BR'
    print('BR')

# Getting the middle pixel values of the tag
Tag_name = ''
if grid_without_padding[1, 1] == 255:
    Tag_name = Tag_name + '1'
else:
    Tag_name = Tag_name + '0'

if grid_without_padding[1, 2] == 255:
    Tag_name = Tag_name + '1'
else:
    Tag_name = Tag_name + '0'

if grid_without_padding[2, 2] == 255:
    Tag_name = Tag_name + '1'
else:
    Tag_name = Tag_name + '0'

if grid_without_padding[2, 1] == 255:
    Tag_name = Tag_name + '1'
else:
    Tag_name = Tag_name + '0'

# Showing them on color the image
image = cv2.putText(image, 'Tag_name =', (0,15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)
image = cv2.putText(image, Tag_name, (115,15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)
image = cv2.putText(image, 'Orientation =', (0,35), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)
image = cv2.putText(image, orientation, (115,35), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)

# Showing the finalized image
cv2.imshow("Image", image)

print(Tag_name)
# print(grid_data)
# print(grid_without_padding)
cv2.waitKey(0)
cv2.destroyAllWindows()
