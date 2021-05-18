# Importing required libraries
import cv2
import numpy as np


def set_points(frame):
    points = np.array([[1, 1]], dtype=np.float32)
    for i in range(0, frame.shape[1], 25):
        for j in range(0, frame.shape[0], 25):
            points = np.append(points, [[i, j]], axis=0)
    points = points.astype('float32')
    return False, points

video_data = cv2.VideoCapture('/home/zeus/Desktop/Cars_On_Highway.mp4')
received_points = True
_, frame = video_data.read()
old_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
lk_param = dict(winSize=(45, 45),
                maxLevel=1,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
result = cv2.VideoWriter('part1_a.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         1, (int(video_data.get(3)), int(video_data.get(4))))


while video_data.isOpened():  # Loop is active till last frame of the Video
    is_success, frame = video_data.read()  # Read each frame of video & return if fetched and each frame
    if is_success is False:  # if the frame was not fetched successfully (i.e. Last frame)
        break  # Exit the while loop
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.medianBlur(frame, 7)
    if received_points:
        received_points, points = set_points(frame)
    new_points, status, error = cv2.calcOpticalFlowPyrLK(old_frame, frame, points, None, **lk_param)
    old_frame = frame
    for i in range(len(new_points)):
        points_ = tuple(points[i])
        new_points_ = tuple(new_points[i])
        diff_o = int(abs(points_[0]-new_points_[0])*1.25)
        diff_i = int(abs(points_[1]-new_points_[1])*1.25)
        new_points__ = (int(new_points_[0] + diff_o), int(new_points_[1] + diff_i))
        frame = cv2.arrowedLine(frame, points_, new_points__, (0, 0, 255), 2, tipLength=0.5)
    result.write(frame)
    frame = cv2.resize(frame, (1850, 950))
    cv2.imshow("bk", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
result.release()
video_data.release()