import numpy as np
import cv2
import scipy.sparse as sp


def tsmooth(img, lam=0.01, sigma=3.0, sharpness=0.001):
    # img is the image nparray whose value is in the range[0, 1]
    wx, wy = compute_texture_weights(img, sigma, sharpness)
    s = solve_linear_equation(img, wx, wy, lam)
    return s


def compute_texture_weights(img, sigma, sharpness):
    dt0_v = np.vstack((np.diff(img, axis=0), img[0, :, np.newaxis] - img[-1, :, np.newaxis]))
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
    ch = img.shape[2]
    k = r * c
    dx = -lam * wx.reshape((k, 1))
    dy = -lam * wy.reshape((k, 1))
    tempx = np.hstack((wx[:, -1, np.newaxis], wx[:, -2, np.newaxis]))
    tempy = np.vstack((wy[-1, :, np.newaxis], wy[-2, :, np.newaxis]))
    dxa = -lam * tempx.reshape((tempx.shape[0] * tempx.shape[1]))
    dya = -lam * tempy.reshape((tempy.shape[0] * tempy.shape[1]))
    tempx = np.hstack((wx[:, -1, np.newaxis], np.zeros((img.shape[0], img.shape[1] - 1))))
    tempy = np.vstack((wy[-1, :, np.newaxis], np.zeros((img.shape[0] - 1, img.shape[1]))))
    dxd1 = -lam * tempx.reshape((tempx.shape[0] * tempx.shape[1], 1))
    dyd1 = -lam * tempy.reshape((tempy.shape[0] * tempy.shape[1], 1))
    wx[:, -1] = 0
    wy[-1, :] = 0
    dxd2 = -lam * wx.reshape((wx.shape[0] * wx.shape[1], 1))
    dyd2 = -lam * wy.reshape((wy.shape[0] * wy.shape[1], 1))

    Ax = sp.spdiags(np.hstack((dxd1, dxd2), np.array([-k + r, -r]), k, k)).toarray()
    Ay = sp.spdiags(np.vstack((dyd1, dyd2), np.array([-r + 1, -1]), k, k)).toarray()

    D = 1 - (dx + dy + dxa + dya)
    A = (Ax + Ay) + (Ax + Ay).transpose() + sp.spdiags(D, 0, k, k).toarray()

    output = img
    for i in range(0, ch):
        tin = img[:, :, i, np.newaxis]
        tout = A / tin
        output[:, :, i] = tout.reshape((r, c))
    return output
