import enhancement
import cv2
import pprint
import numpy as np


if __name__ == '__main__':
    img_path = 'images/sun_set.jpg'
    img = cv2.imread(img_path)
    img = img / 255.0
    print 'Shape is {}'.format(img.shape)
    print 'Range is {}'.format(np.max(img))
    pprint.pprint(img[:, :, 0])
