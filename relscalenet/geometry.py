import numpy as np


def to_homogeneous(x):
    # x is (..., N, 2) or (..., N, 3)
    pts_shape = x.shape[:-1]
    h_coord = np.ones((*pts_shape, 1))
    return np.concatenate([x, h_coord], axis=-1)


def from_homogeneous(x):
    return x[..., 0:-1] / x[..., -1:]


def calibrate_pts(pts, K):
    pts_calib = pts.copy()
    pts_calib[:,0] -= K[0,2]
    pts_calib[:,1] -= K[1,2]
    pts_calib[:,0] /= K[0,0]
    pts_calib[:,1] /= K[1,1]
    return pts_calib


def uncalibrate_pts(pts, K):
    pts = to_homogeneous(pts)
    pts = K @ pts.T
    pts = from_homogeneous(pts.T)
    return pts


def triangulate(R,t,x1,x2):
    num_pts = x1.shape[0]
    x1h = np.c_[ x1, np.ones(num_pts) ]
    x2h = np.c_[ x2, np.ones(num_pts) ]
    X = []
    for i in range(num_pts):
        z1 = x1h[i,:]
        z2 = x2h[i,:]

        S1 = np.array([[0, -z1[2], z1[1]],
                     [z1[2], 0, -z1[0]],
                     [-z1[1], z1[0], 0]])
        S2 = np.array([[0, -z2[2], z2[1]],
                     [z2[2], 0, -z2[0]],
                     [-z2[1], z2[0], 0]])

        A = np.r_[ np.c_[S1, np.zeros(3)],  S2 @ np.c_[R, t] ]

        U, S, Vh = np.linalg.svd(A, full_matrices=True)
        X.append(Vh[-1,0:3].transpose() / Vh[-1,3])

    return np.array(X)


def essential_from_pose(R,t):
    t = t.flatten()
    return np.array([[0, -t[2], t[1]],
                     [t[2], 0, -t[0]],
                     [-t[1], t[0], 0]]) @ R


def sampson_error(F,x1,x2):
    num_pts = x1.shape[0]
    x1h = np.c_[ x1, np.ones(num_pts) ]
    x2h = np.c_[ x2, np.ones(num_pts) ]
    Fx1 = F @ x1h.transpose()
    Fx2 = F.transpose() @ x2h.transpose()

    C = np.sum(x2h.transpose() * Fx1, axis=0)
    denom = Fx1[0,:]**2 + Fx1[1,:]**2 + Fx2[0,:]**2 + Fx2[1,:]**2

    samp_err = np.abs(C) / np.sqrt(denom)
    return samp_err


def rotation_angle(R):
    return np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def angle(v1,v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs
