import numpy as np
import traceback

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordiantes of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camara intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels
    """

    ##### STUDENT CODE START #####

    Pc_homo = np.hstack((pixels, np.ones((pixels.shape[0], 1))))
    R_cw = np.linalg.inv(R_wc)
    t_cw = -np.matmul(R_cw,t_wc)

    # # R_inv = np.hstack((R_cw[:, :-1], t_cw[:, None]))
    R_inv = np.hstack((R_cw[:, 0][:, None], R_cw[:, 1][:, None], t_cw[:, None]))
    Pw = np.zeros((Pc_homo.shape[0], 3))

    for i in range(pixels.shape[0]):
        Pw[i,:] = np.transpose(np.linalg.inv(K@R_inv) @ Pc_homo[i,:].T)
        Pw[i,:] = Pw[i,:] / Pw[i,-1]
        Pw[i,-1] = 0

    ##### STUDENT CODE END #####
    return Pw
