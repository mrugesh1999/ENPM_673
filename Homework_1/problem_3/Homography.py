# Importing Required libraries
import numpy as np

# Setting suppress as True to make sure NP arrays are shown as an integer
np.set_printoptions(suppress=True)


# Defining the function to calculate and return homography matrix
def homography_8x9(x1, x2, x3, x4, y1, y2, y3, y4, xp1, xp2, xp3, xp4, yp1, yp2, yp3, yp4):
    # Creating a matrix to feed into matrix A
    MatrixA = np.zeros((8, 9))

    # Setting values in matrix A based on the given parameters
    MatrixA[0][0] = -x1
    MatrixA[0][1] = -y1
    MatrixA[0][2] = -1
    MatrixA[0][6] = x1 * xp1
    MatrixA[0][7] = y1 * xp1
    MatrixA[0][8] = xp1

    MatrixA[1][3] = -x1
    MatrixA[1][4] = -y1
    MatrixA[1][5] = -1
    MatrixA[1][6] = x1 * yp1
    MatrixA[1][7] = y1 * yp1
    MatrixA[1][8] = yp1

    MatrixA[2][0] = -x2
    MatrixA[2][1] = -y2
    MatrixA[2][2] = -1
    MatrixA[2][6] = x2 * xp2
    MatrixA[2][7] = y2 * xp2
    MatrixA[2][8] = xp2

    MatrixA[3][3] = -x2
    MatrixA[3][4] = -y2
    MatrixA[3][5] = -1
    MatrixA[3][6] = x2 * yp2
    MatrixA[3][7] = y2 * yp2
    MatrixA[3][8] = yp2

    MatrixA[4][0] = -x3
    MatrixA[4][1] = -y3
    MatrixA[4][2] = -1
    MatrixA[4][6] = x3 * xp3
    MatrixA[4][7] = y3 * xp3
    MatrixA[4][8] = xp3

    MatrixA[5][3] = -x3
    MatrixA[5][4] = -y3
    MatrixA[5][5] = -1
    MatrixA[5][6] = x3 * yp3
    MatrixA[5][7] = y3 * yp3
    MatrixA[5][8] = yp3

    MatrixA[6][0] = -x4
    MatrixA[6][1] = -y4
    MatrixA[6][2] = -1
    MatrixA[6][6] = x4 * xp4
    MatrixA[6][7] = y4 * xp4
    MatrixA[6][8] = xp4

    MatrixA[7][3] = -x4
    MatrixA[7][4] = -y4
    MatrixA[7][5] = -1
    MatrixA[7][6] = x4 * yp4
    MatrixA[7][7] = y4 * yp4
    MatrixA[7][8] = yp4

    # Using self made SVD function to calculate U, Sigma, and VT
    U, S, VT = svd(MatrixA)

    # Accessing the last column of VT (Corresponding to least sigma value)
    Homography = VT[8]

    # Reshaping it to the form asked (3x3)
    Homography = np.reshape(Homography, (3, 3))

    # Print the final matrix derived
    print('Homography Matrix')
    print(Homography)

    # Returning the matrix as return parameter
    return Homography


# Defining the SVD function to calculate Singular Value Decomposition
def svd(matrix_a):
    # Calculating AT * A
    AT_A = np.dot(np.transpose(matrix_a), matrix_a)

    # Eigenvalues of AT * A = VT
    # returning eigen values and vectors
    eig_val_V, eig_vec_V = np.linalg.eig(AT_A)
    sorted_val = eig_val_V.argsort()[::-1]

    # sorted all eigenvectors as largest eigen values come first
    new_eig_vec_V = eig_vec_V[:, sorted_val]

    # The final VT matrix from V
    new_eig_vec_VT = np.transpose(new_eig_vec_V)

    # Calculating A * AT
    A_AT = np.dot(matrix_a, np.transpose(matrix_a))

    # U is simply A * AT
    eig_val_U, eig_vec_U = np.linalg.eig(A_AT)

    # Sorting the U matrix as well
    sorted_val1 = eig_val_U.argsort()[::-1]
    new_eig_val_U = eig_val_U[sorted_val1]
    new_eig_vec_U = eig_vec_U[:, sorted_val1]

    # Adding values of sigma in the diagonal matrix temporarily
    diagonal_matt = np.diag((np.sqrt(new_eig_val_U)))

    # The final sigma will be of shape matrix_a
    Sigma_final = np.zeros_like(matrix_a).astype(np.float64)
    Sigma_final[:diagonal_matt.shape[0], :diagonal_matt.shape[1]] = diagonal_matt

    # Return the parameters U, S, VT
    return new_eig_vec_U, Sigma_final, new_eig_vec_VT


# Calling the function to find homography with given parameters
Homography_Matrix = homography_8x9(5, 150, 150, 5, 5, 5, 150, 150, 100, 200, 220, 100, 100, 80, 80, 200)
