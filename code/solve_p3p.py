import numpy as np
from itertools import combinations

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Invoke Procrustes function to find R, t
    # You may need to select the R and t that could transoform all 4 points correctly. 
    # R,t = Procrustes(Pc_3d, Pw[1:4])
    ##### STUDENT CODE END #####

    a = np.linalg.norm( Pw[2, :] - Pw[3, :] )
    b = np.linalg.norm( Pw[1, :] - Pw[3, :] )
    c = np.linalg.norm( Pw[1, :] - Pw[2, :] )

    caliberated_camera_points = (np.linalg.inv(K) @ np.hstack((Pc, np.ones((4, 1)))).T).T

    j = np.array([ (1 / np.linalg.norm(caliberated_camera_points[k, :])) * caliberated_camera_points[k, :] for k in range(0, 4) ])

    cosine_alpha = j[2] @ j[3]
    cosine_beta = j[1] @ j[3]
    cosine_gamma = j[1] @ j[2]

    A4 = ((a**2 - c**2) / b**2 - 1)**2 - 4 * (c**2 / b**2) * cosine_alpha**2
    A3 = 4 * (((a**2 - c**2) / b**2) * (1 - ((a**2 - c**2) / b**2)) * cosine_beta - (1 - ((a**2 + c**2) / b**2)) * cosine_alpha * cosine_gamma + 2 * (c**2 / b**2) * (cosine_alpha**2) * cosine_beta)
    A2 = 2 * ( ((a**2 - c**2) / b**2)**2 - 1 + 2 * ((a**2 - c**2) / b**2)**2 * cosine_beta**2 + 2 * ((b**2 - c**2) / b**2) * cosine_alpha**2 - 4 * ((a**2 + c**2) / b**2) * cosine_alpha * cosine_beta * cosine_gamma + 2 * ((b**2 - a**2) / b**2) * cosine_gamma**2 )
    A1 = 4 * ( -((a**2 - c**2) / b**2) * (1 + ((a**2 - c**2) / b**2)) * cosine_beta + 2 * (a**2 / b**2) * (cosine_gamma**2) * cosine_beta - (1 - ((a**2 + c**2) / b**2)) * cosine_alpha * cosine_gamma )
    A0 = (1 + (a**2 - c**2) / b**2)**2 - 4 * (a**2 / b**2) * cosine_gamma**2

    A = np.array([A4, A3, A2, A1, A0])

    v_roots = np.roots(A)
    v_roots = v_roots[np.isreal(v_roots)]
    v_roots = v_roots[v_roots > 0]
    v_roots = np.real(v_roots)

    u_roots = ((-1 + (a**2 - c**2) / b**2) * np.square(v_roots) - 2 * ((a**2 - c**2) / b**2) * cosine_beta * v_roots + 1 + ((a**2 - c**2) / b**2)) / (2 * (cosine_gamma - v_roots * cosine_alpha))
    u_roots = u_roots[u_roots > 0]

    R_best = None
    t_best = None
    min_error = float('inf')
    for i in range(len(v_roots)):

        s_1 = np.sqrt((b ** 2) / (1 + v_roots[i]**2 - 2 * v_roots[i] * cosine_beta))
        
        s_2 = u_roots[i] * s_1
        
        s_3 = v_roots[i] * s_1

        p = np.vstack(( s_1*j[1,:], s_2*j[2,:], s_3*j[3,:] ))
        Rp, tp = Procrustes(Pw[1:, :], p)

        Pc_3d = K @ (Rp @ Pw[0, :].T + tp)
        Pc_3d = (Pc_3d / Pc_3d[-1])[:-1]
       
        error = np.linalg.norm(Pc_3d - Pc[0, :])

        if error < min_error:
            min_error = error
            R_best = Rp
            t_best = tp

    return np.linalg.inv(R_best), (-np.linalg.inv(R_best) @ t_best)

def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    ##### STUDENT CODE START #####

    X_centroid = np.mean(X, axis = 0)
    Y_centroid = np.mean(Y, axis = 0)

    X_prime = (X - X_centroid).T
    Y_prime = (Y - Y_centroid).T

    R_hat = Y_prime.dot(X_prime.T)

    [U, _, Vt] = np.linalg.svd(R_hat)
    V = Vt.T

    d = np.linalg.det(V.dot(U.T))
    S = np.eye(3, dtype=np.float64)
    S[-1, -1] =  S[-1, -1] * d

    R = U.dot(S.dot(Vt))

    t = Y_centroid - R.dot(X_centroid)

    ##### STUDENT CODE END #####

    return R, t
