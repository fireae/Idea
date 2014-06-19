# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:50:21 2014

@author: Administrator
"""

import cv2 as cv
import numpy as np
import os


def matchImage1(img, tpl):
    img_h, img_w= img.shape
    tpl_h, tpl_w = tpl.shape
    print img.shape
    print tpl.shape
    
    if (img_w < tpl_w) or (img_h < tpl_h):
        return
    sum = 0
    sum_min = 1000000000000000
    idx_row = 0
    idx_col = 0
    for row in range(img_h - tpl_h):
        for col in range(img_w - tpl_w):
            for i in range(tpl_h):
                for j in range(tpl_w):
                    if (tpl[i, j]):
                       sum = sum + np.abs(tpl[i, j] - img[row+i, col+j]) 
            print row, col, sum
            
            if sum < sum_min:
                sum_min = sum
                idx_row = row
                idx_col = col
            sum = 0
    
    print sum_min, idx_row, idx_col
    cv.rectangle(img, (idx_row, idx_col), (idx_row+tpl_h, idx_col+tpl_w), (255, 255, 255), 5)
    cv.imshow("result", img)
    cv.waitKey(0)
    print (idx_row, idx_col), (idx_row+tpl_h, idx_col+tpl_w)
    return sum_min, idx_row, idx_col
    
def matchImage(img, tpl):
    result = cv.matchTemplate(img, tpl, cv.TM_SQDIFF_NORMED)
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(result)
    val, result = cv.threshold(result, 0.01, 0, cv.THRESH_TOZERO)
    result8 = cv.normalize(result,None,0,255,cv.NORM_MINMAX,cv.CV_8U)
    cv.rectangle(img, minLoc, (10, 10), (255, 255, 255))
    cv.imshow("result", img)
    cv.waitKey(0)
    print minVal, maxVal, minLoc, maxLoc

if __name__ == '__main__':
    path_img = 'E:\\workplace\\python\\123061\\290.bmp'
    path_tpl = 'E:\\workplace\\python\\123061\\26 .bmp'
    img = cv.imread(path_img, 1)
    tpl = cv.imread(path_tpl, 1)
    img2 = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    tpl2 = cv.cvtColor(tpl, cv.COLOR_RGB2GRAY)
    matchImage1(img2, tpl2)
  