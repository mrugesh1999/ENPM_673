# Importing required libraries
import cv2
import numpy as np
from collections import Counter


# Creating a function to warp perspective
def warp_perspective(H, image, H_max, W_max):
    """
    :param H: Homography matrix (3X3)
    :param image: Source ima
    :param H_max: Max height of destination
    :param W_max: Max width of destination
    :return: warped image
    """

    try:
        H_inv = np.linalg.inv(H)
        warped_ = np.zeros((H_max, W_max, 3), np.uint8)
        for a in range(H_max):
            for b in range(W_max):
                f = [a, b, 1]
                f = np.reshape(f, (3, 1))
                x, y, z = np.matmul(H_inv, f)
                warped_[a][b] = image[int(y / z)][int(x / z)]
        return warped_
    except:
        pass


# Creating a function to find homography
def find_homography(point1, point2):
    """
    :param point1: list of points (source)
    :param point2: list of points (destination
    :return: Homography matrix
    """
    try:
        A = []
        for i in range(0, len(point1)):
            x, y = point1[i][0], point1[i][1]
            u, v = point2[i][0], point2[i][1]
            A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
            A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
        A = np.asarray(A)
        U, S, Vh = np.linalg.svd(A)  # Using SVD file
        H_col = Vh[-1, :] / Vh[-1, -1]
        H = H_col.reshape(3, 3)
        return H
    except:
        pass


# Getting the user input
print("Choose Tag video to put tag on!!")
print("press 0 for Tag0")
print("press 1 for Tag1")
print("press 2 for Tag2")
print("press 3 for multiple tags")

ent = int(input("Your input: "))

# If statements to choose the video based on user input
if ent == 0:
    video_data = cv2.VideoCapture('Tag0.mp4')
    result = cv2.VideoWriter('Tag0.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, (int(video_data.get(3)), int(video_data.get(4))))
elif ent == 1:
    video_data = cv2.VideoCapture('Tag1.mp4')
    result = cv2.VideoWriter('Tag1.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, (int(video_data.get(3)), int(video_data.get(4))))
elif ent == 2:
    video_data = cv2.VideoCapture('Tag2.mp4')
    result = cv2.VideoWriter('Tag2.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, (int(video_data.get(3)), int(video_data.get(4))))
elif ent == 3:
    video_data = cv2.VideoCapture('multipleTags.mp4')
    result = cv2.VideoWriter('multipleTags.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, (int(video_data.get(3)), int(video_data.get(4))))
else:
    print("No Tags! Try again")
    exit(0)


# Initiating the process by going frame by frame
while video_data.isOpened():  # loop will run till last frame

    img_1 = cv2.imread('/home/zeus/Desktop/testudo.png', 1)     # Sourcing the testudo Image
    is_success, frame = video_data.read()  # Accessing each frame

    if is_success is False:  # is_success is a flag which will set to false on last frame or in case of fetching error
        break

    found_ctr = []      # List of found contours initiated

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # Converting to grayscale
    ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)      # Converting to binary image

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)    # Detecting contours

    false_contours = []     # Creating a list to detect false contours

    # Using tree hierarchy to detect contour with no parent and no child
    for i, h in enumerate(hierarchy[0]):
        if h[2] == -1 or h[3] == -1:    # The last 2 element of the hierarchy matrix
            false_contours.append(i)

    # If the contour is not in false contour, then it is a true contour
    contours = [c for i, c in enumerate(contours) if i not in false_contours]

    # sorting them and accessing the first three contours
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

    # The final filter is to converting it to approximate rectangle
    final_contours = []

    # Accessing each contour individually
    for c in sorted_contours:
        epsilon = cv2.arcLength(c, True)       # Finding the epsilon
        approx = cv2.approxPolyDP(c, epsilon * .015, True)      # Multiplying epsilon with calibrated constant
        if len(approx) == 4:    # If the polygon has 4 sides it is a rectangle
            final_contours.append(approx)       # Appending them to final list

    # Detecting the corners of the contour
    corners = []
    for points in final_contours:
        corner_coordinates = []
        for point in points:    # Accessing each corner point in rectangle
            corner_coordinates.append([point[0][0], point[0][1]])
        corners.append(corner_coordinates)      # Adding those four points list to a list
        src_pts = corner_coordinates        # Feeding them to source
        dst_pts = [[0, 0], [0, 500], [500, 500], [500, 0]]      # Feeding the destination size (Image size)
        H = find_homography(np.float32(src_pts), np.float32(dst_pts))       # Finding homography from SRC to DST
        warped = warp_perspective(H, frame, 500, 500)       # Warping the image for bird-eye view
        grid_data = np.zeros((8, 8))    # Creating a grid
        pixel_list = []
        img = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)      # Converting it to gray image
        ret, thresh = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)   # Converting it to binary image
        img = thresh

        # Creating the array to find rotation
        for i in range(8):
            for j in range(8):
                list_of_val = []
                for k in range(62):
                    for l in range(62):
                        list_of_val.append(img[int((62.5 * i) + k), int((62.5 * j) + l)])   # Accessing each pixel value
                # Calculating MOde
                data = Counter(list_of_val)
                get_mode = dict(data)
                mode = [k for k, v in get_mode.items() if v == max(list(data.values()))]
                if len(mode) == len(list_of_val):
                    grid_data[i, j] = 0
                else:
                    grid_data[i, j] = mode[0]

        # Removing padding
        grid_without_padding = grid_data[2:6, 2:6]

        # Rotating source image based on  Tag orientation
        if grid_without_padding[0, 0] >= 200:
            img_1 = cv2.rotate(img_1, cv2.ROTATE_180)
            print("TL")
        elif grid_without_padding[0, 3] >= 200:
            img_1 = cv2.rotate(img_1, cv2.ROTATE_90_COUNTERCLOCKWISE)
            print("TR")
        elif grid_without_padding[3, 0] >= 200:
            img_1 = cv2.rotate(img_1, cv2.ROTATE_90_CLOCKWISE)
            print("BL")
        elif grid_without_padding[3, 3] >= 200:
            print("BR")

        H_inv = np.linalg.inv(H)    # Inverse homography

        # Creating mask
        Mask_new = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
        cv2.fillPoly(Mask_new, [np.int32(src_pts)], (255, 255, 255))
        Mask_new = cv2.bitwise_not(Mask_new)
        Final = cv2.bitwise_and(frame, frame, mask=Mask_new)

        # Warping back image on masked frame
        for a in range(0, 500):
            for b in range(0, 500):
                x_test, y_test, z_test = np.dot(H_inv, [a, b, 1])
                frame[int(y_test / z_test)][int(x_test / z_test)] = img_1[a][b]     # Changing pixel vals
    cv2.imshow("final", frame)
    result.write(frame)     # Saving result to result variable
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_data.release()
result.release()
