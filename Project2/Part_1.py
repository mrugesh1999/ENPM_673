# Importing libraries
import cv2
import numpy as np

# Sourcing video
video = cv2.VideoCapture('Night Drive - 2689.mp4')

# Setting up output video
out = cv2.VideoWriter('Problem1_cl_gma.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (900, 600))

# looping through each frame
while True:
    success, frame = video.read()
    # If this is not the last frame
    if success:
        # Converting to HSV from BGR image
        frame2hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Accessing the V value from HSV image
        hsv_v = frame2hsv[:, :, 2]

        # finding the Contrast Limited Adaptive Histogram Equalization on V value
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(15, 15))
        frame_clahe = clahe.apply(hsv_v)

        # Setting gamma value, based on the video
        gamma = 1.5
        invGamma = 1.0 / gamma

        # Creatin a look
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
        frame_gamma = cv2.LUT(frame_clahe.astype(np.uint8), table.astype(np.uint8))
        frame2hsv[:, :, 2] = frame_gamma

        # converting back from HSV to BGR format
        frame_improved = cv2.cvtColor(frame2hsv, cv2.COLOR_HSV2BGR)

        # showing the image
        frame_improved = cv2.resize(frame_improved, (900, 600))

        cv2.imshow('clahe', frame_clahe)
        cv2.imshow('gamma correction', frame_clahe)
        out.write(frame_improved)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # if ret is False release the video which will exit the loop
    else:
        video.release()
        print("End Of Video")
        break
out.release()
cv2.destroyAllWindows()