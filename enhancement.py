import cv2
import numpy as np
from tsmooth import tsmooth
import scipy.optimize as so


def enhance(img_path, k=None, mu=0.5, a=-0.3293, b=1.1258, lam=0.5, sigma=5):
    img = cv2.imread(img_path) / 255.0
    t_b = np.max(img, axis=2)
    downsample = cv2.resize(t_b, None, fx=0.5, fy=0.5)
    tsm = tsmooth(downsample, lam, sigma)
    t_our = cv2.resize(tsm, (img.shape[0], img.shape[1]))

    if k is not None:
        corrected = apply_correction(img, k, a, b)
        corrected = np.min(corrected, 1)
    else:
        is_poor = np.argwhere(t_our < 0.5)[0]
        corrected = max_entropy_enhance(img, is_poor)

    t = np.stack((t_our, t_our, t_our), axis=2)
    W = np.power(t, mu)

    I2 = img * W
    J2 = corrected * (1 - W)

    fused = I2 + J2
    return fused


def apply_correction(img, k, a, b):
    beta = np.exp((1 - np.power(k, a))*b)
    gamma = np.power(k, a)
    J = np.power(img, gamma) * beta
    return J


def max_entropy_enhance(img, is_poor, k, a, b):
    Y = reg2gm(np.max(cv2.resize(img, (50, 50)), 0))
    is_poor = reg2gm(cv2.resize(is_poor, (50, 50)))

    def correct_entropy(k_factor):
        p = np.histogram(apply_correction(img, k_factor, a, b), bins=np.array(range(0, 256)))
        entropy = -np.sum(p * np.log2(p))
        return entropy
    opt_k = so.fminbound(correct_entropy, 1, 7)
    J = apply_correction(img, opt_k, a, b) - 0.01
    return J





def reg2gm(img):
    if img.shape[2] == 3:
        img = np.power(img[:, :, 0] * img[:, :, 1] * img[:, :, 2], 1/3)
    return img