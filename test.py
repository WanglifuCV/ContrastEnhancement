import enhancement
import cv2
import pprint
import numpy as np
from enhancement import enhance


if __name__ == '__main__':
    img_path = 'images/lowlight-landscape.jpg'
    img = cv2.imread(img_path)
    fused = enhance(img_path)
