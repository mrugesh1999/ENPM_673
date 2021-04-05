# Importing libraries
import cv2
import numpy as np

# Sourcing video
cap = cv2.VideoCapture('Night Drive - 2689.mp4')

# Setting up output video
out = cv2.VideoWriter('Problem1_gamma_corrected.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (900, 600))

# looping through each frame
while True:
    success, frame = cap.read()
    # If this is not the last frame
    if success:
        # Converting to HSV from BGR image
        frame2hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Accessing the V value from HSV image
        hsv_v = frame2hsv[:, :, 2]

        # Setting gamma value, based on the video
        gamma = 1.5
        invGamma = 1.0 / gamma

        # Creatin a look
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
        frame_gamma = cv2.LUT(hsv_v.astype(np.uint8), table.astype(np.uint8))
        frame2hsv[:, :, 2] = frame_gamma

        # converting back from HSV to BGR format
        frame_final = cv2.cvtColor(frame2hsv, cv2.COLOR_HSV2BGR)

        # showing the image
        frame_final = cv2.resize(frame_final, (900, 600))

        out.write(frame_final)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # if ret is False release the video which will exit the loop
    else:
        cap.release()
        print("End Of Video")
        break
out.release()
cv2.destroyAllWindows()
