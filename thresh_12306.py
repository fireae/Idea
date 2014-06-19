# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 11:44:13 2014

@author: Administrator
"""

import cv2 as cv
import numpy as np
import os

def thresh(src):
    print src
    img = cv.imread(src, 0)
    retval, dst = cv.threshold(img, 100, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
#    cv.namedWindow('a')
#    cv.imshow('a', dst)
#    cv.waitKey(0)
    save_name = src[0:-4] + '.jpg'
    cv.imwrite(save_name, dst)
    
def listimage(path):
    print path
    files = os.listdir(path)
    for f in files:
        thresh(os.path.join(path, f))

if __name__ == '__main__':
    path = 'E:\\workplace\\python\\123061'
    listimage(path)
        