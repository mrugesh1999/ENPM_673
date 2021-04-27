import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
from PIL import Image


def sift_detector(imgA, imgB):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(imgA, None)
    kp2, des2 = sift.detectAndCompute(imgB, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    best_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
            best_matches.append([m])

    feature_1 = []
    feature_2 = []

    for i, match in enumerate(good):
        feature_1.append(kp1[match.queryIdx].pt)
        feature_2.append(kp2[match.trainIdx].pt)
    return feature_1, feature_2, kp1, kp2, best_matches


def fundamental_matrix(feat_1, feat_2):
    feat_1_mean_x = np.mean(feat_1[:, 0])
    feat_1_mean_y = np.mean(feat_1[:, 1])
    feat_2_mean_x = np.mean(feat_2[:, 0])
    feat_2_mean_y = np.mean(feat_2[:, 1])
    feat_1[:, 0] = feat_1[:, 0] - feat_1_mean_x
    feat_1[:, 1] = feat_1[:, 1] - feat_1_mean_y
    feat_2[:, 0] = feat_2[:, 0] - feat_2_mean_x
    feat_2[:, 1] = feat_2[:, 1] - feat_2_mean_y
    s_1 = np.sqrt(2.) / np.mean(np.sum((feat_1) ** 2, axis=1) ** (1 / 2))
    s_2 = np.sqrt(2.) / np.mean(np.sum((feat_2) ** 2, axis=1) ** (1 / 2))
    T_a_1 = np.array([[s_1, 0, 0], [0, s_1, 0], [0, 0, 1]])
    T_a_2 = np.array([[1, 0, -feat_1_mean_x], [0, 1, -feat_1_mean_y], [0, 0, 1]])
    T_a = np.dot(T_a_1, T_a_2)
    T_b_1 = np.array([[s_2, 0, 0], [0, s_2, 0], [0, 0, 1]])
    T_b_2 = np.array([[1, 0, -feat_2_mean_x], [0, 1, -feat_2_mean_y], [0, 0, 1]])
    T_b = np.dot(T_b_1, T_b_2)
    x1 = (feat_1[:, 0].reshape((-1, 1))) * s_1
    y1 = (feat_1[:, 1].reshape((-1, 1))) * s_1
    x2 = (feat_2[:, 0].reshape((-1, 1))) * s_2
    y2 = (feat_2[:, 1].reshape((-1, 1))) * s_2
    A = np.hstack((x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones((len(x1), 1))))
    A = np.array(A)
    U, S, V = np.linalg.svd(A)
    V = V.T
    sol = V[:, -1]
    F = sol.reshape((3, 3))
    U_F, S_F, V_F = np.linalg.svd(F)
    S_F[2] = 0
    S_new = np.diag(S_F)
    F_new = U_F @ S_new @ V_F
    F_norm = T_b.T @ F_new @ T_a
    F_norm = F_norm / F_norm[-1, -1]
    return F_norm


def estimate_fundamental_matrix(feature_1, feature_2):
    threshold = 0.5
    max_num_inliers = 0
    F_matrix_best = []
    p = 0.99
    N = np.inf
    count = 0
    while count < N:
        inliers_count = 0
        feature_1_rand = []
        feature_2_rand = []
        random = np.random.randint(len(feature_1), size=8)
        for i in random:
            feature_1_rand.append(feature_1[i])
            feature_2_rand.append(feature_2[i])
        F = fundamental_matrix(np.array(feature_1_rand), np.array(feature_2_rand))
        ones = np.ones((len(feature_1), 1))
        x1 = np.hstack((feature_1, ones))
        x2 = np.hstack((feature_2, ones))
        e1 = np.dot(x1, F.T)
        e2 = np.dot(x2, F)
        error = np.sum(e2 * x1, axis=1, keepdims=True) ** 2 / np.sum(np.hstack((e1[:, :-1], e2[:, :-1])) ** 2, axis=1,
                                                                     keepdims=True)
        inliers = error <= threshold
        inliers_count = np.sum(inliers)

        if inliers_count > max_num_inliers:
            max_num_inliers = inliers_count
            F_matrix_best = F
            # Iterations to run the RANSAC for every frame
        inlier_ratio = inliers_count / len(feature_1)

        if np.log(1 - (inlier_ratio ** 8)) == 0:
            continue

        N = np.log(1 - p) / np.log(1 - (inlier_ratio ** 8))
        count += 1
    return F_matrix_best


def essential_matrix(F, K):
    E = K.T @ F @ K
    U, S, V = np.linalg.svd(E)
    S_new = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    E_new = U @ S_new @ V

    return E_new


def extract_camera_pose(E, K):
    U, D, V = np.linalg.svd(E)
    V = V.T
    W = np.reshape([0, -1, 0, 1, 0, 0, 0, 0, 1], (3, 3))
    C_1 = U[:, 2]
    R_1 = U @ W @ V.T
    C_2 = -U[:, 2]
    R_2 = U @ W @ V.T
    C_3 = U[:, 2]
    R_3 = U @ W.T @ V.T
    C_4 = -U[:, 2]
    R_4 = U @ W.T @ V.T

    if np.linalg.det(R_1) < 0:
        R_1 = -R_1
        C_1 = -C_1
    if np.linalg.det(R_2) < 0:
        R_2 = -R_2
        C_2 = -C_2
    if np.linalg.det(R_3) < 0:
        R_3 = -R_3
        C_3 = -C_3
    if np.linalg.det(R_4) < 0:
        R_4 = -R_4
        C_4 = -C_4

    C_1 = C_1.reshape((3, 1))
    C_2 = C_2.reshape((3, 1))
    C_3 = C_3.reshape((3, 1))
    C_4 = C_4.reshape((3, 1))

    return [R_1, R_2, R_3, R_4], [C_1, C_2, C_3, C_4]


def depth_from_disparity(img):
    img_return = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_return[i, j] = (int(base*focal)/img[i, j])
    return img_return


def stereo_match(left_img, right_img, kernel, max_offset):
    left_img = Image.open(left_img).convert('L')
    left = np.asarray(left_img)
    right_img = Image.open(right_img).convert('L')
    right = np.asarray(right_img)
    w, h = left_img.size
    depth = np.zeros((w, h), np.uint8)
    depth.shape = h, w
    kernel_half = int(kernel / 2)
    offset_adjust = 255 / max_offset
    for y in tqdm(range(kernel_half, h - kernel_half)):
        print(".", end="", flush=True)
        for x in range(kernel_half, w - kernel_half):
            best_offset = 0
            prev_ssd = 65534
            for offset in range(max_offset):
                ssd = 0
                ssd_temp = 0
                for v in range(-kernel_half, kernel_half):
                    for u in range(-kernel_half, kernel_half):
                        ssd_temp = int(left[y + v, x + u]) - int(right[y + v, (x + u) - offset])
                        ssd += ssd_temp * ssd_temp
                if ssd < prev_ssd:
                    prev_ssd = ssd
                    best_offset = offset
            depth[y, x] = best_offset * offset_adjust
    Image.fromarray(depth).save('depth.png')


dataset = int(input("Enter the dataset you want to excute: (enter 1, 2 or 3)"))

if dataset == 1:
    K1 = [5299.313, 0, 1263.818, 0, 5299.313, 977.763, 0, 0, 1]
    K2 = [5299.313, 0, 1438.004, 0, 5299.313, 977.763, 0, 0, 1]
    K1 = np.reshape(K1, (3, 3))
    K2 = np.reshape(K2, (3, 3))
    base = 177.288
    focal = 5299.313

if dataset == 2:
    K1 = [4396.869, 0, 1353.072, 0, 4396.869, 989.702, 0, 0, 1]
    K2 = [4396.869, 0, 1538.86, 0, 4396.869, 989.702, 0, 0, 1]
    K1 = np.reshape(K1, (3, 3))
    K2 = np.reshape(K2, (3, 3))
    base = 144.049
    focal = 4396.869

if dataset == 3:
    K1 = [5806.559, 0, 1429.219, 0, 5806.559, 993.403, 0, 0, 1]
    K2 = [5806.559, 0, 1543.51, 0, 5806.559, 993.403, 0, 0, 1]
    K1 = np.reshape(K1, (3, 3))
    K2 = np.reshape(K2, (3, 3))
    base = 174.019
    focal = 5806.559

else:
    print("Enter either 1 or 2 or 3 and nothing else !!")

sterio = int(input("Do you want to get disparity and depth image ? (1:Yes, 2:No)"))


img1 = cv2.imread('Dataset_3/im0.png')
img2 = cv2.imread('Dataset_3/im1.png')

h1 = img1.shape[0]
w1 = img1.shape[1]
ch1 = img1.shape[2]

h2 = img2.shape[0]
w2 = img2.shape[1]
ch2 = img2.shape[2]

feature_1, feature_2, kp1, kp2, best_matches = sift_detector(img1, img2)
img4 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, best_matches, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite("Before rectification.png", img4)
Best_F_matrix = estimate_fundamental_matrix(feature_1, feature_2)
print("Best Fundamental Matrix:")
print(Best_F_matrix)

E_matrix = essential_matrix(Best_F_matrix, K1)
R, T = extract_camera_pose(E_matrix, K1)
print('R:')
print(R)
print('T')
print(T)
H = []
I = np.array([0, 0, 0, 1])
for i, j in zip(R, T):
    h = np.hstack((i, j))
    h = np.vstack((h, I))
    H.append(h)
print('H:\n', H)

_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(feature_1), np.float32(feature_2), Best_F_matrix,
                                          imgSize=(w1, h1))
print("H1:\n", H1)
print("H2:\n", H2)

img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))
feature_1, feature_2, kp1, kp2, best_matches = sift_detector(img1_rectified, img2_rectified)
cv2.imwrite("rectified_1.png", img1_rectified)
cv2.imwrite("rectified_2.png", img2_rectified)
distances = {}
for j in range(len(feature_1)):
    new_list = np.array([feature_1[j][0], feature_2[j][1], 1])
    new_list = np.reshape(new_list, (3, 1))
    new_list_2 = np.array([feature_1[j][0], feature_2[j][1], 1])
    new_list_2 = np.reshape(new_list_2, (1, 3))
    distances[j] = abs(new_list_2 @ Best_F_matrix @ new_list)

distances_sorted = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}

distances_list = []
v_list = []
for k, v in distances_sorted.items():
    if v[0][0] < 0.05:
        distances_list.append(v[0][0])
        v_list.append(k)

len_distance_list = len(distances_list)

len_distance_list = min(len_distance_list, 30)  # 30 might have to be changed

inliers_1 = []
inliers_2 = []

for x in range(len_distance_list):
    inliers_1.append(feature_1[v_list[x]])
    inliers_2.append(feature_2[v_list[x]])

img3 = cv2.drawMatchesKnn(img1_rectified, kp1, img2_rectified, kp2, best_matches, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite("after rectification.png", img3)
cv2.imwrite('left_img.png', img1_rectified)
cv2.imwrite('right_img.png', img2_rectified)
if sterio == 1:
    img1_rectified = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    img2_rectified = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
    stereo_match(img1_rectified, img2_rectified, 20, 16)
    image = cv2.imread('depth.png', 0)
    colormap = plt.get_cmap('inferno')
    heatmap = (colormap(image) * 2 ** 16).astype(np.uint16)[:, :, :3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    cv2.imwrite("heat_map.png", heatmap)
    image_depth = depth_from_disparity(image)
    image_depth = ((image_depth - image_depth.min()) * (1 / (image_depth.max() - image_depth.min()) * 255)).astype('uint8')
    cv2.imwrite("Image_depth.png", image_depth)
    cv2.imshow("Image_depth", image_depth)
    image = cv2.imread('Image_depth.png', 0)
    colormap = plt.get_cmap('inferno')
    heatmap = (colormap(image) * 2 ** 16).astype(np.uint16)[:, :, :3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    cv2.imwrite("heat_map_depth.png", heatmap)
    cv2.waitKey()
