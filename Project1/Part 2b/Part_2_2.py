# Importing required libraries
import cv2
import numpy as np


def warp_perspective(H, img, maxHeight, maxWidth):
    """
    :param H: Homography from img to 500x500 size image (3x3)
    :param img: Tag frame
    :param maxHeight: Destination Height
    :param maxWidth: Destination Width
    :return: Warped image
    """
    try:
        H_inv = np.linalg.inv(H)
        warped_img = np.zeros((maxHeight, maxWidth, 3), np.uint8)
        for a in range(maxHeight):
            for b in range(maxWidth):
                f = [a, b, 1]
                f = np.reshape(f, (3, 1))
                x, y, z = np.matmul(H_inv, f)
                warped_img[a][b] = img[int(y / z)][int(x / z)]
        return warped_img
    except:
        pass


def projection_matrix(h, K):
    """
    :param h: Homography matrix (3x3)
    :param K: Intrinsic parameter of camera
    :return: Projection matrix
    """
    h1 = h[:, 0]  # taking column vectors h1,h2 and h3
    h2 = h[:, 1]
    h3 = h[:, 2]
    # calculating lambda
    lambda_ = 2 / (np.linalg.norm(np.matmul(np.linalg.inv(K), h1)) + np.linalg.norm(np.matmul(np.linalg.inv(K), h2)))
    BT = lambda_ * np.matmul(np.linalg.inv(K), h)

    # check if determinant is greater than 0 ie. has a positive determinant when object is in front of camera
    determinant = np.linalg.det(BT)

    if determinant > 0:
        b = BT
    else:  # else make it positive
        b = -1 * BT

    r1 = b[:, 0]
    r2 = b[:, 1]  # extract rotation and translation vectors
    r3 = np.cross(r1, r2)

    t = b[:, 2]
    rotation = np.column_stack((r1, r2, r3, t))
    projection_mat = np.matmul(K, rotation)
    return projection_mat


def find_homography(p1, p2):
    """
    :param p1: Destination point
    :param p2: Source point
    :return: Homography matrix (3x3)
    """
    try:
        A = []
        for i in range(0, len(p1)):
            x, y = p1[i][0], p1[i][1]
            u, v = p2[i][0], p2[i][1]
            A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
            A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
        A = np.asarray(A)
        U, S, Vh = np.linalg.svd(A)  # Using SVD file
        L = Vh[-1, :] / Vh[-1, -1]
        H = L.reshape(3, 3)
        return H
    except:
        pass


# Defining intrinsic parameters from question
K = np.array([[1406.08415449821, 0, 0],
              [2.20679787308599, 1417.99930662800, 0],
              [1014.13643417416, 566.347754321696, 1]])

K = K.T     # Setting the transpose of it

# Getting the user input
print("Choose Tag video to put cube on!!")
print("press 0 for Tag0")
print("press 1 for Tag1")
print("press 2 for Tag2")
print("press 3 for multiple tags")

ent = int(input("Your input: "))

# If statements to choose the video based on user input
if ent == 0:
    video_data = cv2.VideoCapture('/home/zeus/Desktop/Tag0.mp4')
    result = cv2.VideoWriter('Tag0.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, (int(video_data.get(3)), int(video_data.get(4))))
elif ent == 1:
    video_data = cv2.VideoCapture('/home/zeus/Desktop/Tag1.mp4')
    result = cv2.VideoWriter('Tag1.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, (int(video_data.get(3)), int(video_data.get(4))))
elif ent == 2:
    video_data = cv2.VideoCapture('/home/zeus/Desktop/Tag2.mp4')
    result = cv2.VideoWriter('Tag2.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, (int(video_data.get(3)), int(video_data.get(4))))
elif ent == 3:
    video_data = cv2.VideoCapture('/home/zeus/Desktop/multipleTags.mp4')
    result = cv2.VideoWriter('multipleTags.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, (int(video_data.get(3)), int(video_data.get(4))))
else:
    print("No Tags! Try again")
    exit(0)

# Initiating the process by going frame by frame
while video_data.isOpened():  # loop will run till last frame

    img_1 = cv2.imread('/home/zeus/Desktop/testudo.png', 1)  # Sourcing the testudo Image
    is_success, frame = video_data.read()  # Accessing each frame

    if is_success is False:  # is_success is a flag which will set to false on last frame or in case of fetching error
        break

    found_ctr = []  # List of found contours initiated

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converting to grayscale
    ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)  # Converting to binary image

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # Detecting contours

    false_contours = []  # Creating a list to detect false contours

    # Using tree hierarchy to detect contour with no parent and no child
    for i, h in enumerate(hierarchy[0]):
        if h[2] == -1 or h[3] == -1:  # The last 2 element of the hierarchy matrix
            false_contours.append(i)

    # If the contour is not in false contour, then it is a true contour
    contours = [c for i, c in enumerate(contours) if i not in false_contours]

    # sorting them and accessing the first three contours
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

    # The final filter is to converting it to approximate rectangle
    final_contours = []

    # Accessing each contour individually
    for c in sorted_contours:
        epsilon = cv2.arcLength(c, True)  # Finding the epsilon
        approx = cv2.approxPolyDP(c, epsilon * .015, True)  # Multiplying epsilon with calibrated constant
        if len(approx) == 4:  # If the polygon has 4 sides it is a rectangle
            final_contours.append(approx)  # Appending them to final list

    # Detecting the corners of the contour
    corners = []
    for points in final_contours:
        corner_coordinates = []
        for point in points:  # Accessing each corner point in rectangle
            corner_coordinates.append([point[0][0], point[0][1]])
        corners.append(corner_coordinates)  # Adding those four points list to a list
        src_pts = corner_coordinates  # Feeding them to source
        dst_pts = [[0, 0], [0, 500], [500, 500], [500, 0]]  # Feeding the destination size (Image size)
        H_cube = find_homography(dst_pts, src_pts)
        P = projection_matrix(H_cube, K)
        
        # Getting points in 3D from projection matrix
        x1, y1, z1 = np.matmul(P, [0, 0, 0, 1])
        x2, y2, z2 = np.matmul(P, [0, 500, 0, 1])
        x3, y3, z3 = np.matmul(P, [500, 0, 0, 1])
        x4, y4, z4 = np.matmul(P, [500, 500, 0, 1])
        x5, y5, z5 = np.matmul(P, [0, 0, -500, 1])
        x6, y6, z6 = np.matmul(P, [0, 500, -500, 1])
        x7, y7, z7 = np.matmul(P, [500, 0, -500, 1])
        x8, y8, z8 = np.matmul(P, [500, 500, -500, 1])
        
        # Drawing lines based on homogeneous points received
        # Top of the cube
        cv2.line(frame, (int(x1 / z1), int(y1 / z1)), (int(x5 / z5), int(y5 / z5)), (255, 0, 0), 2)
        cv2.line(frame, (int(x2 / z2), int(y2 / z2)), (int(x6 / z6), int(y6 / z6)), (255, 0, 0), 2)
        cv2.line(frame, (int(x3 / z3), int(y3 / z3)), (int(x7 / z7), int(y7 / z7)), (255, 0, 0), 2)
        cv2.line(frame, (int(x4 / z4), int(y4 / z4)), (int(x8 / z8), int(y8 / z8)), (255, 0, 0), 2)

        # Bottom of the cube
        cv2.line(frame, (int(x1 / z1), int(y1 / z1)), (int(x2 / z2), int(y2 / z2)), (0, 255, 0), 2)
        cv2.line(frame, (int(x1 / z1), int(y1 / z1)), (int(x3 / z3), int(y3 / z3)), (0, 255, 0), 2)
        cv2.line(frame, (int(x2 / z2), int(y2 / z2)), (int(x4 / z4), int(y4 / z4)), (0, 255, 0), 2)
        cv2.line(frame, (int(x3 / z3), int(y3 / z3)), (int(x4 / z4), int(y4 / z4)), (0, 255, 0), 2)

        # Sides of the cube
        cv2.line(frame, (int(x5 / z5), int(y5 / z5)), (int(x6 / z6), int(y6 / z6)), (0, 0, 255), 2)
        cv2.line(frame, (int(x5 / z5), int(y5 / z5)), (int(x7 / z7), int(y7 / z7)), (0, 0, 255), 2)
        cv2.line(frame, (int(x6 / z6), int(y6 / z6)), (int(x8 / z8), int(y8 / z8)), (0, 0, 255), 2)
        cv2.line(frame, (int(x7 / z7), int(y7 / z7)), (int(x8 / z8), int(y8 / z8)), (0, 0, 255), 2)

    # Showing the video to usee
    cv2.imshow("CUBE VIDEO", frame)
    result.write(frame)     # Saving the result in result variable

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
