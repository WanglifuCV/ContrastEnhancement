import enhancement
import cv2
import pprint
import numpy as np
from enhancement import enhance


if __name__ == '__main__':
    img_name = 'lowlight-landscape.jpg'
    img_path = 'images/' + img_name
    fused_path = 'result/enhanced_' + img_name
    img = cv2.imread(img_path)
    fused = enhance(img_path)
    # cv2.imshow('orig', img)
    # cv2.imshow('enhanced', fused)
    # cv2.waitKey(0)
    cv2.imwrite(fused_path, fused * 255)
