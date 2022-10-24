import numpy as np

def est_homography(X, Y):
    """
    Calculates the homography H of two planes such that Y ~ H*X
    If you want to use this function for hw5, you need to figure out
    what X and Y should be.
    Input:
        X: 4x2 matrix of (x,y) coordinates
        Y: 4x2 matrix of (x,y) coordinates
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. Y ~ H*X

    """

    ##### STUDENT CODE START #####

    A = np.zeros((8, 9))
    for i in range(4):
        a_x = np.array([-X[i, 0], -X[i, 1], -1., 0., 0., 0., X[i, 0] * Y[i, 0], X[i, 1] * Y[i, 0], Y[i, 0]], dtype=np.float64)
        a_y = np.array([0., 0., 0., -X[i, 0], -X[i, 1], -1., X[i, 0] * Y[i, 1], X[i, 1] * Y[i, 1], Y[i, 1]], dtype=np.float64)
        A[2 * i, :],  A[2 * i + 1, :] = a_x,  a_y

    [_, _, Vt] = np.linalg.svd(A)

    # print(f"Matrix V: {Vt[-1, :].reshape(9,1).shape}")

    H = Vt[-1, :].reshape(3, 3)

    H = H / H[-1,-1]

    ##### STUDENT CODE END #####

    return H
