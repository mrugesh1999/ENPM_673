import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
from PIL import Image

K1 = [5299.313, 0, 1263.818, 0, 5299.313, 977.763, 0, 0, 1]
K2 = [5299.313, 0, 1438.004, 0, 5299.313, 977.763, 0, 0, 1]
K1 = np.reshape(K1, (3, 3))
K2 = np.reshape(K2, (3, 3))


def stereo_match(left_img, right_img, kernel, max_offset):
    # Load in both images, assumed to be RGBA 8bit per channel images
    left_img = Image.open(left_img).convert('L')
    left = np.asarray(left_img)
    right_img = Image.open(right_img).convert('L')
    right = np.asarray(right_img)
    w, h = left_img.size  # assume that both images are same size

    # Depth (or disparity) map
    depth = np.zeros((w, h), np.uint8)
    depth.shape = h, w

    kernel_half = int(kernel / 2)
    offset_adjust = 255 / max_offset  # this is used to map depth map output to 0-255 range

    for y in tqdm(range(kernel_half, h - kernel_half)):
        print(".", end="", flush=True)  # let the user know that something is happening (slowly!)

        for x in range(kernel_half, w - kernel_half):
            best_offset = 0
            prev_ssd = 65534

            for offset in range(max_offset):
                ssd = 0
                ssd_temp = 0

                # v and u are the x,y of our local window search, used to ensure a good
                # match- going by the squared differences of two pixels alone is insufficient,
                # we want to go by the squared differences of the neighbouring pixels too
                for v in range(-kernel_half, kernel_half):
                    for u in range(-kernel_half, kernel_half):
                        # iteratively sum the sum of squared differences value for this block
                        # left[] and right[] are arrays of uint8, so converting them to int saves
                        # potential overflow, and executes a lot faster
                        ssd_temp = int(left[y + v, x + u]) - int(right[y + v, (x + u) - offset])
                        ssd += ssd_temp * ssd_temp

                        # if this value is smaller than the previous ssd at this block
                # then it's theoretically a closer match. Store this value against
                # this block..
                if ssd < prev_ssd:
                    prev_ssd = ssd
                    best_offset = offset

            # set depth output for this x,y location to the best match
            depth[y, x] = best_offset * offset_adjust

    # Convert to PIL and save it
    Image.fromarray(depth).save('depth.png')


def fundamental_matrix(feat_1, feat_2):
    # compute the centroids
    feat_1_mean_x = np.mean(feat_1[:, 0])
    feat_1_mean_y = np.mean(feat_1[:, 1])
    feat_2_mean_x = np.mean(feat_2[:, 0])
    feat_2_mean_y = np.mean(feat_2[:, 1])

    # Recenter the coordinates by subtracting mean
    feat_1[:, 0] = feat_1[:, 0] - feat_1_mean_x
    feat_1[:, 1] = feat_1[:, 1] - feat_1_mean_y
    feat_2[:, 0] = feat_2[:, 0] - feat_2_mean_x
    feat_2[:, 1] = feat_2[:, 1] - feat_2_mean_y

    # Compute the scaling terms which are the average distances from origin
    s_1 = np.sqrt(2.) / np.mean(np.sum((feat_1) ** 2, axis=1) ** (1 / 2))
    s_2 = np.sqrt(2.) / np.mean(np.sum((feat_2) ** 2, axis=1) ** (1 / 2))

    # Calculate the transformation matrices
    T_a_1 = np.array([[s_1, 0, 0], [0, s_1, 0], [0, 0, 1]])
    T_a_2 = np.array([[1, 0, -feat_1_mean_x], [0, 1, -feat_1_mean_y], [0, 0, 1]])
    T_a = np.dot(T_a_1, T_a_2)

    T_b_1 = np.array([[s_2, 0, 0], [0, s_2, 0], [0, 0, 1]])
    T_b_2 = np.array([[1, 0, -feat_2_mean_x], [0, 1, -feat_2_mean_y], [0, 0, 1]])
    T_b = np.dot(T_b_1, T_b_2)

    # Compute the normalized point correspondences
    x1 = (feat_1[:, 0].reshape((-1, 1))) * s_1
    y1 = (feat_1[:, 1].reshape((-1, 1))) * s_1
    x2 = (feat_2[:, 0].reshape((-1, 1))) * s_2
    y2 = (feat_2[:, 1].reshape((-1, 1))) * s_2

    # 8-point Hartley
    # A is (8x9) matrix
    A = np.hstack((x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones((len(x1), 1))))

    # Solve for A using SVD
    A = np.array(A)
    U, S, V = np.linalg.svd(A)
    V = V.T

    # last col - soln
    sol = V[:, -1]
    F = sol.reshape((3, 3))
    U_F, S_F, V_F = np.linalg.svd(F)

    # Rank-2 constraint
    S_F[2] = 0
    S_new = np.diag(S_F)

    # Recompute normalized F
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
    # Due to the noise in K, the singular values of E are not necessarily (1, 1, 0)
    # This can be corrected by reconstructing it with (1, 1, 0) singular values
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


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


# Given Projection matrix and point correspondences, estimate 3-D point
def point_3d(pt, pt_, R2, C2, K):
    # Find the projection matrices for respective frames
    C1 = [[0], [0], [0]]
    R1 = np.identity(3)
    R1C1 = -R1 @ C1
    R2C2 = -R2 @ C2
    # Current frame has no Rotation and Translation
    P1 = K @ np.hstack((R1, R1C1))

    # Estimate the projection matrix for second frame using returned R and T values
    P2 = K @ np.hstack((R2, R2C2))
    # P1_T = P1.T
    # P2_T = P2.T
    X = []

    # Solve linear system of equations using cross-product technique, estimate X using least squares technique
    for i in range(len(pt)):
        x1 = pt[i]
        x2 = pt_[i]
        A1 = x1[0] * P1[2, :] - P1[0, :]
        A2 = x1[1] * P1[2, :] - P1[1, :]
        A3 = x2[0] * P2[2, :] - P2[0, :]
        A4 = x2[1] * P2[2, :] - P2[1, :]
        A = [A1, A2, A3, A4]

        U, S, V = np.linalg.svd(A)
        V = V[3]
        V = V / V[-1]
        X.append(V)
    return X


# cheirality condition
def linear_triangulation(pt, pt_, R, C, K):
    # Check if the reconstructed points are in front of the cameras using cheilarity equations
    X1 = point_3d(pt, pt_, R, C, K)
    X1 = np.array(X1)
    count = 0
    # r3(X-C)>0
    for i in range(X1.shape[0]):
        x = X1[i, :].reshape(-1, 1)
        if R[2] @ np.subtract(x[0:3], C) > 0 and x[2] > 0:
            count += 1
    return count


def sift_detector(imgA, imgB):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()  # opencv-contrib-python required
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(imgA, None)
    kp2, des2 = sift.detectAndCompute(imgB, None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
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


def main():
    img1 = cv2.imread('Dataset_3/im0.png')
    img2 = cv2.imread('Dataset_3/im1.png')
    img_ref = cv2.imread('Dataset_1/im1.png')
    # img1 = cv2.resize(img1, (img_ref.shape[1], img_ref.shape[0]), interpolation=cv2.INTER_AREA)
    # img2 = cv2.resize(img2, (img_ref.shape[1], img_ref.shape[0]), interpolation=cv2.INTER_AREA)
    # dim1 = img1.shape
    # dim2 = img2.shape

    h1 = img1.shape[0]
    w1 = img1.shape[1]
    ch1 = img1.shape[2]

    h2 = img2.shape[0]
    w2 = img2.shape[1]
    ch2 = img2.shape[2]

    feature_1, feature_2, kp1, kp2, best_matches = sift_detector(img1, img2)
    Best_F_matrix = estimate_fundamental_matrix(feature_1, feature_2)
    print("Best Fundamental Matrix:", Best_F_matrix)

    E_matrix = essential_matrix(Best_F_matrix, K1)
    R, T = extract_camera_pose(E_matrix, K1)
    print('R:', R)
    print('T', T)
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
    # cv2.imwrite("rectified_1.png", img1_rectified)
    # cv2.imwrite("rectified_2.png", img2_rectified)
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
        # print(v[0][0])
        if v[0][0] < 0.05:  # threshold distance
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


    img1_rectified = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    img2_rectified = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
    img_kernel = np.zeros(shape=(int(h1/8), int(w1/8)))
    for i in range(int(h1/8)):
        for j in range(int(w1/8)):
            kernel = img1_rectified[i*8:(i*8)+8, j*8:(j*8)+8]
            if j < 15:
                image_to_be_scaned = img2_rectified[i*8:(i*8)+8, :1400]
            elif 15 <= j <= int(w1/8) - 25:
                image_to_be_scaned = img2_rectified[i * 8:(i * 8) + 8, (j*8)-15:(j*8)+15]
            elif j > int(w1/8)-25:
                image_to_be_scaned = img2_rectified[i * 8:(i * 8) + 8, w1-1400:]
            res = cv2.matchTemplate(image_to_be_scaned, kernel, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= 0.6)
            max_dif = 20
            final_point = list
            for pt in zip(*loc[::-1]):
                if (pt[1]-(j*8)) < max_dif:
                    final_point = pt
                    max_dif = pt[1]-(j*8)
            img_kernel[i, j] = abs(max_dif)
    new_arr = ((img_kernel - img_kernel.min()) * (1 / (img_kernel.max() - img_kernel.min()) * 255)).astype('uint8')
    print(np.max(new_arr))
    new_arr = cv2.equalizeHist(new_arr)
    cv2.imwrite("disparity.png", new_arr)
    cv2.imwrite("disparity_kernel.png", img_kernel)
    colormap = plt.get_cmap('inferno')
    heatmap = (colormap(new_arr) * 2 ** 16).astype(np.uint16)[:, :, :3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    cv2.imwrite("Heat_map.png", heatmap)
    cv2.imwrite('left_img.png', img1_rectified)
    cv2.imwrite('right_img.png', img2_rectified)
    stereo_match("left_img.png", "right_img.png", 18, 15)

if __name__ == '__main__':
    main()
