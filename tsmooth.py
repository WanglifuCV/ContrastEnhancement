import numpy as np
import cv2
import scipy.sparse as sp
from scipy.sparse.linalg import cg


def tsmooth(img, lam=0.01, sigma=3.0, sharpness=0.001):
    # img is the image nparray whose value is in the range[0, 1]
    wx, wy = compute_texture_weights(img, sigma, sharpness)
    s = solve_linear_equation(img, wx, wy, lam)
    return s


def compute_texture_weights(img, sigma, sharpness):
    dt0_v = np.vstack((np.diff(img, axis=0), img[np.newaxis, 0, :] - img[np.newaxis, -1, :]))
    dt0_h = np.hstack((np.diff(img, axis=1), img[:, 0, np.newaxis] - img[:, -1, np.newaxis]))
    h_kernel = np.ones((1, sigma), np.float32)
    v_kernel = np.ones((sigma, 1), np.float32)
    gauker_v = cv2.filter2D(dt0_v, -1, v_kernel)
    gauker_h = cv2.filter2D(dt0_h, -1, h_kernel)
    w_v = 1.0 / (np.abs(gauker_v) * np.abs(dt0_v) + sharpness)
    w_h = 1.0 / (np.abs(gauker_h) * np.abs(dt0_h) + sharpness)
    return w_h, w_v


def solve_linear_equation(img, wx, wy, lam):
    r = img.shape[0]
    c = img.shape[1]
    if len(img.shape) > 2:
        ch = img.shape[2]
    else:
        ch = 1
    k = r * c
    dx = -lam * wx.reshape((k, 1))
    dy = -lam * wy.reshape((k, 1))
    tempx = np.hstack((wx[:, -1, np.newaxis], wx[:, 0:-1]))
    tempy = np.vstack((wy[np.newaxis, -1, :], wy[0:-1, :]))
    dxa = -lam * tempx.T.reshape((tempx.shape[0] * tempx.shape[1], 1))
    dya = -lam * tempy.T.reshape((tempy.shape[0] * tempy.shape[1], 1))
    tempx = np.hstack((wx[:, -1, np.newaxis], np.zeros((img.shape[0], img.shape[1] - 1))))
    tempy = np.vstack((wy[np.newaxis, -1, :], np.zeros((img.shape[0] - 1, img.shape[1]))))
    dxd1 = -lam * tempx.T.reshape((tempx.shape[0] * tempx.shape[1], 1))
    dyd1 = -lam * tempy.T.reshape((tempy.shape[0] * tempy.shape[1], 1))
    wx[:, -1] = 0
    wy[-1, :] = 0
    dxd2 = -lam * wx.T.reshape((wx.shape[0] * wx.shape[1], 1))
    dyd2 = -lam * wy.T.reshape((wy.shape[0] * wy.shape[1], 1))

    Ax = sp.spdiags(np.hstack((dxd1, dxd2)).T, np.array([-k + r, -r]), k, k)#.toarray()
    Ay = sp.spdiags(np.hstack((dyd1, dyd2)).T, np.array([-r + 1, -1]), k, k)#.toarray()

    D = 1 - (dx + dy + dxa + dya)
    A = (Ax + Ay) + (Ax + Ay).T + sp.spdiags(D.T, 0, k, k)#.toarray()
    
    # L = scipy.sparse.cholmod.cholesky(A)
    output = img
    for i in range(0, ch):
        tin = img.T.reshape((k, 1))
        tout = cg(A, tin, tol=0.001)
        # err = np.linalg.norm(A * tout[0] - tin)
        # print 'Error = {}'.format(err)
        output = tout[0].reshape((r, c), order='F')
    return output
