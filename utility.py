# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 13:54:03 2014

@author: Administrator
"""

import cv2 as cv
import numpy as np
import os
import math

CANNY = 1

def get_points_from_image(img_path, thresh=50, simpleto=100, method=CANNY):
    
    print img_path
    if not os.path.isfile(img_path):
        raise 'image path failed.'
        
    img = cv.imread(img_path, 0)
    if method == CANNY:
        edges = cv.Canny(img, thresh, thresh*3)
   
    A = img.copy()
    px, py = np.gradient(A)
    h, w = img.shape
    print 'w %d, h%d' %(w, h)
    points = []
    for y in range(h):
        for x in range(w):
            c = edges[y, x]
            if c == 255:
                points.append([x, y])

    r = 2
    while len(points) > simpleto:
        newpoints = points
        xr = range(0, w, r)
        yr = range(0, h, r)
        for p in points:
            if p[0] not in xr and p[1] not in yr:
                newpoints.remove(p)
                if len(points) <= simpleto:
                    T = np.zeros((simpleto, 1))
                    for i, (x, y) in enumerate(points):
                        T[i] = math.atan2(py[y, x], px[y, x]) + math.pi /2
                    A = np.zeros((h, w))
                    for pt in points:
                        cv.circle(A, (pt[0], pt[1]), 2, (255, 255, 255))
                    cv.namedWindow('a')
                    cv.imshow('a', A)
                    cv.waitKey(0)
                    return points, np.asmatrix(T)
        r = r + 1
    
    T = np.zeros((simpleto, 1))
    for i, (x, y) in enumerate(points):
        T[i] = math.atan2(py[y, x], px[y, x]) + math.pi/2

    return points, np.asmatrix(T)
    
if __name__ == '__main__':
    path = 'D:\\Downloads\\Python-Shape-Context-master\\Python-Shape-Context-master\\test1.png'
    points, T = get_points_from_image(path)
    print points
    print T
    