# Importing required libraries

import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
import time

# Setting suppress as true will make sure that NP arrays are NOT in exponential terms
np.set_printoptions(suppress=True)

# MKing fonts bit larger for ease of read
plt.rcParams.update({'font.size': 22})


# Defining plot_graph function to plot the graph with given parameters such as Equation, X Axis length & label name
# Type of parameters: Equation = String & x_axis_length = Range & label_str = string


def plot_graph(equation, x_axis_length, label_str, linewidth='1.0'):
    x = np.array(x_axis_length)  # Creating 1D NP array with length of no. of Points
    y = eval(equation)  # Solving equation passed as a String(str)
    plt.plot(x, y, label=label_str, linewidth=linewidth)  # Plotting Graph


# Creating RANSAC as Function
def ransac(mypoints, iteration, thresh):
    # Initializing iteration counter N, and return lists
    N = 0
    best_inliner_ = []
    best_outliner_ = []

    # Loop to apply algorithm N times to the data points (N epochs)
    while N < iteration:

        # Choosing three random values from given data set
        Chosen_Pts = random.sample(range(0, len(mypoints)), 3)
        x1 = mypoints[Chosen_Pts[0]][0]
        x2 = mypoints[Chosen_Pts[1]][0]
        x3 = mypoints[Chosen_Pts[2]][0]
        y1 = mypoints[Chosen_Pts[0]][1]
        y2 = mypoints[Chosen_Pts[1]][1]
        y3 = mypoints[Chosen_Pts[2]][1]

        # Applying the equation to find the parameter constants A, B, C
        A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / ((x1 - x2) * (x1 - x3) * (x2 - x3))
        B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / ((x1 - x2) * (x1 - x3) * (x2 - x3))
        C = y1 - (A * x1 * x1) - (B * x1)

        # Initializing the list of inLiners and outLiners for each epoch
        inLiners = []
        outLiners = []

        # Finding if the point is inside the defined threshold
        for iii in range(len(mypoints)):
            xi = mypoints[iii][0]
            yi = mypoints[iii][1]
            Y_Yi = (A * xi * xi) + (B * xi) + C

            # If it is, then append it to the inLiners list
            if abs(Y_Yi - yi) <= thresh:
                inLiners.append((xi, yi))

            # If not, append to the outLiner list
            else:
                outLiners.append((xi, yi))

            # Checking if the this epoch generates better model or not
            if len(inLiners) > len(best_inliner_):
                # If it does, modify the best parameters and model related data
                best_outliner_ = outLiners
                best_inliner_ = inLiners
                best_parameter_ = [A, B, C]

        # Increment the counter
        N = N + 1

    # Assign the parameter values to constants to return
    alfa_ = best_parameter_[0]
    beta_ = best_parameter_[1]
    gama_ = best_parameter_[2]

    # Uncomment this if you want to print them
    # print(best_inliner_)
    # print(best_outliner_)
    # print(best_parameter_)

    # Returning values derived from this function in int, int, int, list, list format
    return alfa_, beta_, gama_, best_inliner_, best_parameter_


# Using Least Squared (LS) Error for  curve fitting
def least_sqrd(mypoints):
    # Creating a Matrix *for* columns [ Xi^2  Xi  1 ] with dimensions N X 3, where N is number of contours
    Matrix_A = np.ones((len(mypoints), (len(mypoints[0]) + 1)), dtype=int)

    # Updating values as [ Xi^2  Xi  1 ] in Matrix
    for ii in range(len(mypoints)):
        Matrix_A[ii, 0] = (mypoints[ii][0] * mypoints[ii][0])
        Matrix_A[ii, 1] = mypoints[ii][0]

    # Creating a Matrix *for* columns [ Yi ] with dimensions N X 1, where N is number of contour
    Matrix_B = np.ones((len(mypoints), 1), dtype=int)

    # Updating values as [ Yi ] in Matrix
    for j in range(len(mypoints)):
        Matrix_B[j, 0] = mypoints[j][1]

    # Applying Pseudo Inverse of Matrix A in Ax=B with x = (pseudo inverse (A) * B)
    Matrix_X = np.dot((np.linalg.inv(np.dot(np.transpose(Matrix_A), Matrix_A))),
                      np.dot(np.transpose(Matrix_A), Matrix_B))

    # Make sure that the Matrix X is NOT integer(int)
    Matrix_X = Matrix_X.astype('float')

    # print(Matrix_X)

    # Extracting Values of each constants
    aa = Matrix_X[0]
    bb = Matrix_X[1]
    cc = Matrix_X[2]

    # Printing the constants
    # print(a, b, c)
    return aa, bb, cc


def TLS(mypoints):
    # Creating list to extract elements from my points.
    xi = []
    yi = []
    xi_pow_2 = []

    # Using for loop to fill the data into the list
    for i in range(len(mypoints)):
        xi_pow_2.append(mypoints[i][0] * mypoints[i][0])
        xi.append(mypoints[i][0])
        yi.append(mypoints[i][1])

    # Initializing constants to calculate total
    xi_pow_2_total = 0
    yi_total = 0
    xi_total = 0

    # Using for loop to get the sum
    for i in range(len(mypoints)):
        xi_pow_2_total = xi_pow_2_total + xi_pow_2[i]
        xi_total = xi_total + xi[i]
        yi_total = yi_total + yi[i]

    # Calculating mean
    xi_mean = xi_total / len(mypoints)
    yi_mean = yi_total / len(mypoints)
    xi_pow_2_mean = xi_pow_2_total / len(mypoints)

    # Creating matrix with required shape
    matrix_a = np.ones((len(mypoints), 3))

    # Update values in matrix
    for i in range(len(mypoints)):
        matrix_a[i][0] = xi_pow_2[i] - xi_pow_2_mean
        matrix_a[i][1] = xi[i] - xi_mean
        matrix_a[i][2] = yi[i] - yi_mean

    # Using SVD to calculate just the V matrix (NOT  transposed)
    # Calculating AT*A
    AT_A = np.dot(np.transpose(matrix_a), matrix_a)

    # Calculating eigen values and eigen vector for AT*A
    eig_val_V, eig_vec_V = np.linalg.eig(AT_A)

    # Getting indices based on descending order
    sorted_val = eig_val_V.argsort()[::-1]

    # Modifying the eigen vector matrix based on indices values
    new_eig_vec_V = eig_vec_V[:, sorted_val]

    # Choosing the last column of V
    parameters = new_eig_vec_V[:, 2]

    # Accessing parameters individually from the matrix parameter
    aye = parameters[0] / parameters[2]
    bee = parameters[1] / parameters[2]
    cee = parameters[2]

    # Calculating d from the equation
    dee = (parameters[0] * xi_pow_2_mean) + (parameters[1] * xi_mean) + (parameters[2] * yi_mean)
    d = dee / cee

    # Returning derived values from functions
    return aye, bee, d


# Sourcing the video from the source
# Pass the path to video as a String(str)
video_data = cv2.VideoCapture('/home/zeus/Desktop/Perception/Ball_travel_10fps.mp4')

# Creating an empty list with name myPoints to store object location data
myPoints = []

# Creating a while loop to go through video frame by frame
# for implementing algorithm and to get the location data of the object
# as well as storing location data to myPoints list and drawing
while video_data.isOpened():  # Loop is active till last frame of the Video
    is_success, frame = video_data.read()  # Read each frame of video & return if fetched and each frame
    if is_success is False:  # if the frame was not fetched successfully (i.e. Last frame)
        break  # Exit the while loop

    # Converting to HSV to use find contours function
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Defining the HSV values' range for Red color in HSV format
    # Red color has HSV value ranges for Hue [0 to 10] and for this object range [170 to 180] worked for hue
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])

    # Creating a mask to use find contours
    mask = cv2.inRange(hsv, lower_red, upper_red)
    # cv2.imshow("Mask", mask)                    # This show the masked red object as white

    # Using RETR_EXTERNAL flag as object has no child contours
    # Using CHAIN_APPROX_NONE as we want all the points of contours, not just edges, to find the centroid
    # Here there is no parent-child relationship of contours, thus hierarchy is not important
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Below is the line of code to use for drawing the contour
    # cv2.drawContours(frame, contours, -1, (255, 0, 0), 3)

    # Finding centroid of each contour every frame.
    for con in contours:
        # Getting the moment of each contour (con) in contour list
        # The green's formula is used to find the moments
        M = cv2.moments(con)

        # The center of mass moments are defined as
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Appending these numbers to initially created list myPoints
        myPoints.append((cX, cY))

        # Using for loop to create points at centroid of each object
        for points in myPoints:
            # cv2.FILLED can be used as well, but increasing thickness of circle (Point Circle) works as well.
            cv2.circle(frame, (points[0], points[1]), 12, (0, 255, 0), -1)

    # Creating Delay so it is easy to examine
    time.sleep(0.1)
    # Playing video frame by frame at each iteration of loop
    cv2.imshow("Object Path", frame)

    # If pressed q on keyboard, it will exit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

a, b, c = least_sqrd(myPoints)
# Plotting the graph for LS error line fitting solution curve
plot_graph('-1 * ((a*(x**2))+(b*x)+c)', range(0, 2350), 'LS', linewidth='1.0')

# Plotting the scatter plot of the centroids in each frame
for i in range(len(myPoints)):
    plt.scatter(myPoints[i][0], (-1 * myPoints[i][1]), c='black', marker='o', edgecolors='black')

# Using RANSAC function to get the best-fit model
alfa, beta, gama, best_inliner, best_parameter = ransac(myPoints, 2500, 35)

# Plotting graph based on RANSAC
plot_graph('-1 * ((alfa*(x**2))+(beta*x)+gama)', range(0, 2350), 'RANSAC')

# Applying TLS to my points and getting constants
aye, bee, d = TLS(myPoints)

# Plotting graph with derived parameters
plot_graph('(aye*(x**2))+(bee*x)-d', range(0, 2350), 'TLS')
# Initialize the labels and showing the final plot
plt.legend()
plt.show()
