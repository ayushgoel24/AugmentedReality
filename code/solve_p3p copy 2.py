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

    print(f"Pc = \n{Pc}\n")
    print(f"Pw = \n{Pw}\n")

    a = np.linalg.norm( Pw[2, :] - Pw[3, :] )
    b = np.linalg.norm( Pw[1, :] - Pw[3, :] )
    c = np.linalg.norm( Pw[1, :] - Pw[2, :] )

    u = Pc[:, 0]
    v = Pc[:, 1]

    caliberated_camera_points = (np.linalg.inv(K) @ np.hstack((u[:, None], v[:, None], np.ones((4, 1)))).T).T

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

    roots = np.roots(A)
    roots = roots[np.isreal(roots)]
    roots = roots[roots > 0]

    roots = np.real(roots)
    print(f"roots = \n{roots}\n")

    u = ((-1 + (a**2 - c**2) / b**2) * np.square(roots) - 2 * ((a**2 - c**2) / b**2) * cosine_beta * roots + 1 + ((a**2 - c**2) / b**2)) / 2 * (cosine_gamma - roots * cosine_alpha)
    u = u[u>0]
    print(f"u = \n{u}\n")

    # s_1 = np.sqrt((c ** 2) / (1 + np.square(u) - 2 * u * cosine_gamma))

    # s_2 = np.multiply(u, s_1)
    # s_3 = np.multiply(roots, s_1)

    # s = np.hstack((s_1[:, None], s_2[:, None], s_3[:, None]))

    best_R = None
    best_t = None
    min_error = float('inf')
    for i in range(len(u)):
        s_1 = np.sqrt((c ** 2) / (1 + np.square(u[i]) - 2 * u[i] * cosine_gamma))
        s_2 = u[i] * s_1
        s_3 = roots[i] * s_1

        p = np.vstack((s_1*j[1,:], s_2*j[2,:], s_3*j[3,:]))
        R, t = Procrustes(Pw[1:, :], p)

        # R = np.linalg.inv(R)
        # t = (-np.linalg.inv(R) @ t)
        # P = K @ np.hstack((R, t[:, None])) @ np.vstack((Pw[0, None].T, np.ones((1,1))))

        P = K @ (R @ Pw[0, :].T + t) #snp.vstack((R.T, -R.T@t.T)).T @ np.hstack((Pw[0], [1]))
        print(f"P = \n{P}\n")

        # P = K @ np.hstack((np.linalg.inv(R), (-np.linalg.inv(R) @ t)[:, None])) @ np.vstack((Pw[0, None].T, np.ones((1,1))))
        error = np.linalg.norm(P - Pw[0].T)
        print(f"error = \n{error}\n")

        if error < min_error:
            min_error = error
            best_R = R
            best_t = t

    return best_R, best_t

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
