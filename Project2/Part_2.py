# Importing required libraries
import numpy as np
from scipy import signal as sg
import cv2
import time

# Initializing variables
set_selected = int
dest_dim = (500, 500)
initial_frame_num = 0
count = 0
crp_w = dest_dim[0]
crp_h = dest_dim[1]
hist_peak_r = int
hist_peak_l = int
left_lane_slope = int
right_lane_slope = int
initial_frame_num = 0


def read_each_frame(frame_no):
    """
    This function will read each frame and fetch the requested frame
    :param frame_no: Int
    :return:    frame: The requested frame
                success: If NOT last frame, return False
    """
    success = bool
    if set_selected == 1:
        frame = cv2.imread("data_1/data/" + ('0000000000' + str(frame_no))[-10:] + '.png')      # Sourcing frame
        if frame is None:           # If frame not found
            success = False         # Return variable as false

    if set_selected == 2:
        video = cv2.VideoCapture("data_2/challenge_video.mp4")      # Sourcing video
        video.set(1, frame_no)                                      # Sourcing frame_no
        success, frame = video.read()                               # Reading that frame and storing it to variable
    return success, frame


def pre_process(frame_given):
    """
    Returns Undistorted image
    :param frame_given: The frame matrix
    :return: Processed image
    """
    undistorted_image = cv2.undistort(frame_given, K, dist)     # Undistorted the image using dist and k matrix given
    blurred = cv2.GaussianBlur(undistorted_image, (7, 7), 1)    # Blurring image for noise reduction
    return blurred


def warp_image(src_image, H, height, width):
    """
    Warps source image by applying given homography matrix
    :param src_image: Source Image
    :param H: Homography matrix
    :param height: Height of the destination Image
    :param width: Width of the destination Image
    :return: Warped Image
    """
    indice_y, indice_x = np.indices((height, width), dtype=np.float32)  # Get indices of given Height and Width matrix

    # Finding a matrix with last column as 1 and first two as the X and Y indices respectively
    line_homography_indices = np.array([indice_x.ravel(), indice_y.ravel(), np.ones_like(indice_x).ravel()])

    # Applying homography multiplication to each point
    mapped_indice = H.dot(line_homography_indices)

    # Converting from homogeneous coordinates to cartesian coordinates
    mapped_x, mapped_y = mapped_indice[:-1] / mapped_indice[-1]

    # Mapping the X and Y points to a (Height x Width) dimensions
    maped_x = mapped_x.reshape(height, width).astype(np.float32)
    maped_y = mapped_y.reshape(height, width).astype(np.float32)

    # lookup these points to the destination
    warped_image = cv2.remap(src_image, maped_x, maped_y, cv2.INTER_LINEAR)
    return warped_image


def cvt_binary(given_image):
    """
    Converts the given Image to the binary Image
    :param given_image: BGR Image
    :return: Binary Image based on selected threshold
    """
    # Convert the given image to the HSV format from BGR
    hsv_img = cv2.cvtColor(given_image, cv2.COLOR_BGR2HSV)

    # Thresholding the image based on parameters set
    thresholded_list = []
    for thresh in threshold:
        thresholded_list.append(cv2.inRange(hsv_img, thresh[0], thresh[1]))

    # Checking if the image has two threshold or not
    if len(thresholded_list) != 1:
        hsv_binary_image = thresholded_list[0]      # This will be the case for data set 2
        hsv_binary_image = cv2.bitwise_or(hsv_binary_image, thresholded_list[0])
        hsv_binary_image = cv2.bitwise_or(hsv_binary_image, thresholded_list[1])

    else:
        hsv_binary_image = thresholded_list[0]      # This will be the case for data set 1

    return hsv_binary_image, thresholded_list


def peak_detect(hsv_binary_image):
    """
    Detects peak of histogram of given binary image
    :param hsv_binary_image: Image matrix
    :return:
    """
    # Initializing variables
    global count
    global hist_peak_r
    global hist_peak_l

    # Creating an array to be used as dimension set to get histogram
    inds = np.nonzero(hsv_binary_image)

    # Generate histogram
    num_pixels, bins = np.histogram(inds[1], bins=crp_w, range=(0, crp_w))

    # Getting peaks in histogram with Continuous Wavelet Transform
    hist_peak_all = sg.find_peaks_cwt(num_pixels, np.arange(1, 50))

    if len(hist_peak_all) == 0:  # No peaks detected
        hist_peak_r = hist_peak_r
        hist_peak_l = hist_peak_l
        right_lane = False
        left_lane = False

    if len(hist_peak_all) == 1:  # one peak detected
        if hist_peak_all[0] >= crp_w / 2 and abs(hist_peak_all[0] - hist_peak_r) < 30:
            hist_peak_r = hist_peak_all[0]
            hist_peak_l = hist_peak_l
            right_lane = True
            left_lane = False
        if hist_peak_all[0] <= crp_w / 2 and abs(hist_peak_all[0] - hist_peak_l) < 30:
            hist_peak_l = hist_peak_all[0]
            hist_peak_r = hist_peak_r
            right_lane = False
            left_lane = True
        else:
            hist_peak_r = hist_peak_r
            hist_peak_l = hist_peak_l
            right_lane = False
            left_lane = False

    # Multiple peaks Detected
    else:
        peak_vals = []
        for peak in hist_peak_all:
            peak_vals.append(num_pixels[peak])

        # Find the two highest value peaks
        max1_ind = peak_vals.index(max(peak_vals))
        temp = peak_vals.copy()
        temp[max1_ind] = 0
        max2_ind = peak_vals.index(max(temp))
        big_hist_peak_all = [hist_peak_all[max1_ind], hist_peak_all[max2_ind]]
        big_hist_peak_all.sort()

        if count == 0:
            hist_peak_l = big_hist_peak_all[0]
            hist_peak_r = big_hist_peak_all[1]
            left_lane = True
            right_lane = True
        else:
            found_hist_peak_l = False
            found_hist_peak_r = False
            for peak in hist_peak_all:
                if abs(peak - hist_peak_l) <= 30:
                    found_hist_peak_l = True
                    hist_peak_l = peak
                if abs(peak - hist_peak_r) <= 30:
                    found_hist_peak_r = True
                    hist_peak_r = peak
            if found_hist_peak_l and found_hist_peak_r:
                left_lane = True
                right_lane = True
            elif found_hist_peak_r:
                right_lane = True
                left_lane = False
                hist_peak_l = hist_peak_l
                left_lane = False
            elif found_hist_peak_l:
                left_lane = True
                right_lane = False
                hist_peak_r = hist_peak_r
                right_lane = False
            else:
                hist_peak_r = hist_peak_r
                hist_peak_l = hist_peak_l
                right_lane = False
                left_lane = False

    hist_peak_r = hist_peak_r
    hist_peak_l = hist_peak_l
    # Returns right peak column, left peak column, if left lane is detected, if right lane is detected,
    # White pixel index, frame number
    return hist_peak_r, hist_peak_l, left_lane, right_lane, inds, count


def find_lanes(hist_peak_l, hist_peak_r, left_lane, right_lane, inds):
    """
    Find the line coefficients
    :param hist_peak_l: Left peak
    :param hist_peak_r: Right Peak
    :param left_lane: Bool if that lane has been detected or not
    :param right_lane: BOol if that lane has been detected or not
    :param inds: Indices of the white pixels
    :return:
    """
    global left_lane_slope
    global right_lane_slope
    pnts_l, pnts_r = [], []

    # Iterating through all white pixel indices
    for i, x in enumerate(inds[1]):
        y = inds[0][i]
        # If it fall within specified range, append it to that line list
        if int(hist_peak_l - 60 // 2) <= x <= int(hist_peak_l + 60 // 2):
            pnts_l.append([x, y])
        elif int(hist_peak_r - 60 // 2) <= x <= int(hist_peak_r + 60 // 2):
            pnts_r.append([x, y])

    # Modify to array to perform polyfit
    pnts_l = np.asarray(pnts_l)
    pnts_r = np.asarray(pnts_r)

    # getting the slope of these lines
    # Fact: np.polyfit uses TLS method by default
    if not right_lane and not left_lane:
        pass
    elif (not left_lane) and len(pnts_r) > 0:
        right_lane_slope = np.polyfit(pnts_r[:, 1], pnts_r[:, 0], 1)
    elif (not right_lane) and len(pnts_l) > 0:
        left_lane_slope = np.polyfit(pnts_l[:, 1], pnts_l[:, 0], 1)
    else:
        if not (len(pnts_l)) == 0:
            left_lane_slope = np.polyfit(pnts_l[:, 1], pnts_l[:, 0], 1)
        if not (len(pnts_r)) == 0:
            right_lane_slope = np.polyfit(pnts_r[:, 1], pnts_r[:, 0], 1)
    # Each slope function will have M and C constants in the list
    return left_lane_slope, right_lane_slope


def overlay_line(left_lane_slope, right_lane_slope, frame):
    """
    Overlay the lines on the image
    :param left_lane_slope: Slope of the left line
    :param right_lane_slope: Slope of the right line
    :param frame: Frame corresponding to that data
    :return:
    """
    # Getting the corners
    x = [left_lane_slope[1], (500 * left_lane_slope[0]) + left_lane_slope[1],
         (500 * right_lane_slope[0]) + right_lane_slope[1], right_lane_slope[1]]
    y = [0, 500, 500, 0]

    lane_image = np.zeros((crp_h, crp_w, 3), np.uint8)
    corners = []
    for i in range(4):
        corners.append((int(x[i]), int(y[i])))

    contour = np.array(corners, dtype=np.int32)
    cv2.drawContours(lane_image, [contour], -1, (150, 100, 150), -1)

    cv2.line(lane_image, corners[0], corners[1], (255, 255, 0), 10)
    cv2.line(lane_image, corners[2], corners[3], (255, 255, 0), 10)

    # Creating an average line by averaging both the lanes
    p1 = (0, (left_lane_slope[1] + right_lane_slope[1]) // 2)
    p2 = (crp_h, (crp_h * left_lane_slope[0] + left_lane_slope[1] +
                        crp_h * right_lane_slope[0] + right_lane_slope[1]) // 2)

    # Slope of the new line is calculated
    lineslope = (p2[1] - p1[1]) / (p2[0] - p1[0])

    lane_image = lane_image
    H_inv = np.linalg.inv(H)
    warped_lane = warp_image(lane_image, H_inv, frame.shape[0], frame.shape[1])

    x[0] -= 5
    x[1] -= 5
    x[2] += 5
    x[3] += 5

    X_s = np.array([x, y, np.ones_like(x)])
    sX_c = H.dot(X_s)
    X_c = sX_c / sX_c[-1]
    corners = []

    for i in range(4):
        corners.append((X_c[0][i], X_c[1][i]))

    contour = np.array(corners, dtype=np.int32)
    lane_overlay_img = frame.copy()
    cv2.drawContours(lane_overlay_img, [contour], -1, (0, 0, 0), -1)
    lane_overlay_img = cv2.bitwise_or(lane_overlay_img, warped_lane)

    overlay = cv2.addWeighted(frame, 0.5, lane_overlay_img, 0.5, 0)

    if -.03 <= lineslope <= .04 :
        text = 'Going straight'
        overlay = cv2.putText(overlay, text, (550, 50), cv2.FONT_HERSHEY_PLAIN,
                              2, (0, 0, 255), 2, cv2.LINE_AA)
    elif lineslope < -.03:
        text = 'Right Turn'
        overlay = cv2.putText(overlay, text, (550, 50), cv2.FONT_HERSHEY_PLAIN,
                              2, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        text = 'Left Turn'
        overlay = cv2.putText(overlay, text, (550, 50), cv2.FONT_HERSHEY_PLAIN,
                              2, (0, 255, 0), 2, cv2.LINE_AA)
    return overlay


# Getting user input
while True:
    set_selected = int(input("Choose data set either 1 or 2 "))
    if 0 < set_selected <= 2:
        break


# Based on user input setting parameters
if set_selected == 1:
    threshold = [[(0, 0, 215), (255, 50, 255)]]     # This is brute forced value
    corners_src = [(590, 275), (710, 275), (940, 515), (155, 515)]      # This defines the ROI
    corners_dest = [(100, 0), (400, 0), (400, 500), (100, 500)]
    K = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02],           # Given data
                  [0.000000e+00, 9.019653e+02, 2.242509e+02],
                  [0.000000e+00, 0.000000e+00, 1.000000e+00]])
    dist = np.array([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])


elif set_selected == 2:
    threshold = [[(0, 56, 100), (255, 255, 255)], [(0, 0, 190), (255, 255, 255)]]
    corners_src = [(610, 480), (720, 480), (960, 680), (300, 680)]
    corners_dest = [(100, 0), (400, 0), (400, 500), (100, 500)]
    K = np.array([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
                  [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist = np.array([-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02])

# Creating column matrix for homography calculation
pnts_src = np.float32(corners_src).reshape(4, 1, 2)
pnts_dest = np.float32(corners_dest).reshape(4, 1, 2)

# Finding Homography
H = cv2.findHomography(pnts_dest, pnts_src)[0]

cur_frame = initial_frame_num
success, frame = read_each_frame(initial_frame_num)
frame = pre_process(frame)

if set_selected ==1 :
    file = 'problem2_dataset' + str(set_selected) + '.mp4'
    out = cv2.VideoWriter(file, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 15, (1392, 512))
if set_selected == 2:
    file = 'problem2_dataset' + str(set_selected) + '.mp4'
    out = cv2.VideoWriter(file, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (1280, 720))

while success:
    # Bird eye view
    bird_eye = warp_image(frame, H, crp_h, crp_w)
    if set_selected == 2:
        bird_eye[:, 200:350] = (0, 0, 0)

    # Prepare the Image, Edge detection, create new image to fill with contours
    hsv_binary_image, hsv_threshs = cvt_binary(bird_eye)
    inds = np.nonzero(hsv_binary_image)
    num_pixels, bins = np.histogram(inds[1], bins=crp_w, range=(0, crp_w))
    image_ = np.zeros((502, 502), dtype='uint8')
    for i in range(500):
        image_[0:num_pixels[i], i] = (255)
    image_ = cv2.flip(image_, 0)
    cv2.imshow("image", image_)
    right_peak, left_peak, found_left_lane, found_right_lane, inds, count = peak_detect(hsv_binary_image)

    # Fit Lines using line fitting based on DetectPeaks()
    left_lane_slope, right_lane_slope = find_lanes(left_peak, right_peak, found_left_lane, found_right_lane, inds)

    # Overlay the Lane and Lane Lines On Original Frame
    overlay = overlay_line(left_lane_slope, right_lane_slope, frame)


    out.write(overlay)
    cv2.imshow("Lane Overlay", overlay)
    cv2.imshow("Bird eye view", bird_eye)
    cv2.imshow("hsv binary", hsv_binary_image)
    if cv2.waitKey(1) == ord('w'):
        time.sleep(4)

    if cv2.waitKey(1) == ord('q'):
        break

    count += 1
    cur_frame += 1
    success, frame = read_each_frame(cur_frame)


out.release()
