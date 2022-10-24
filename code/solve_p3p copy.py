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

    a = np.linalg.norm( Pw[1, :] - Pw[2, :] )
    b = np.linalg.norm( Pw[2, :] - Pw[3, :] )
    c = np.linalg.norm( Pw[3, :] - Pw[1, :] )

    comb = combinations([0, 1, 2, 3], 3)
    for i in list(comb):
        a = np.linalg.norm( Pw[ i[0, :] ] - Pw[ i[1, :] ], ord = 2)
        b = np.linalg.norm( Pw[ i[1, :] ] - Pw[ i[2, :] ], ord = 2)
        c = np.linalg.norm( Pw[ i[2, :] ] - Pw[ i[0, :] ], ord = 2)

        u = np.vstack((Pc[i[0], 0], Pc[i[1], 0], Pc[i[2], 0]))
        v = np.vstack((Pc[i[0], 1], Pc[i[1], 1], Pc[i[2], 1]))

        j = np.array([ np.divide( np.vstack((u[k], v[k], K[0, 0])), np.sqrt(u[k] ** 2 + v[k] ** 2 + K[0, 0] ** 2)) for k in range(0,3) ])

        cosine_alpha = j[1].dot( j[2] )
        cosine_beta = j[0].dot( j[2] )
        cosine_gamma = j[0].dot( j[1] )

        A4 = ((a**2 - c**2) / b**2 - 1)**2 - 4 * (c**2 / b**2) * cosine_alpha**2
        A3 = 4 * (((a**2 - c**2) / b**2) * (1 - ((a**2 - c**2) / b**2)) * cosine_beta - (1 - ((a**2 + c**2) / b**2)) * cosine_alpha * cosine_gamma + 2 * (c**2 / b**2) * (cosine_alpha**2) * cosine_beta)
        A2 = 2 * ( ((a**2 - c**2) / b**2)**2 - 1 + 2 * ((a**2 - c**2) / b**2)**2 * cosine_beta**2 + 2 * ((b**2 - c**2) / b**2) * cosine_alpha**2 - 4 * ((a**2 + c**2) / b**2) * cosine_alpha * cosine_beta * cosine_gamma + 2 * ((b**2 - a**2) / b**2) * cosine_gamma**2 )
        A1 = 4 * ( -((a**2 - c**2) / b**2) * (1 + ((a**2 - c**2) / b**2)) * cosine_beta + 2 * (a**2 / b**2) * (cosine_gamma**2) * cosine_beta - (1 - ((a**2 + c**2) / b**2)) * cosine_alpha * cosine_gamma )
        A0 = (1 + (a**2 - c**2) / b**2)**2 - 4 * (a**2 / b**2) * cosine_gamma**2

        A = np.array([A4, A3, A2, A1, A0])

        roots = np.roots(A)
        roots = roots[np.isreal(roots)]
        roots = roots[roots > 0]

        



    return R, t

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
