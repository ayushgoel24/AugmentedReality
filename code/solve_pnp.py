from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Homography Approach
    # Following slides: Pose from Projective Transformation

    H = est_homography(Pw, Pc)

    H_prime = np.linalg.inv(K) @ H

    h1_prime = H_prime[:, 0]
    h2_prime = H_prime[:, 1]

    h1_prime_cross_h2_prime = np.cross(h1_prime, h2_prime)

    z = np.hstack((h1_prime[:, None], h2_prime[:, None], h1_prime_cross_h2_prime[:, None]))

    [U, S, Vt] = np.linalg.svd(z)

    d = np.linalg.det(U @ Vt)

    s = np.eye(3, dtype=np.float64)
    s[-1, -1] = d

    R = U @ s @ Vt

    t = H_prime[:, 2] / np.linalg.norm(h1_prime)
    ##### STUDENT CODE END #####

    print(np.linalg.inv(R))

    print(-np.linalg.inv(R) @ t)

    return np.linalg.inv(R), -np.linalg.inv(R) @ t
