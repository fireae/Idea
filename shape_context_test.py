# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 15:28:37 2014

@author: Administrator
"""

import utility as uti
import ShapeContext as sc

def get_euclidean_dist(points):
        '''
        compute eculidean distance from each point to all others:
        '''
        row = len(points)
        p_x = []
        p_y = []
        for i in range(row):
            p_x.append(points[i][0])
            p_y.append(points[i][1])
        
        p_x_tile = np.tile(p_x, (row, 1))
        p_y_tile = np.tile(p_y, (row, 1))
        p_x_tile_t = p_x_tile.transpose()
        p_y_tile_t = p_y_tile.transpose()
        
        p_x_pow2 = (p_x_tile - p_x_tile_t) **2
        p_y_pow2 = (p_y_tile - p_y_tile_t) **2
        dist = np.sqrt(p_x_pow2+p_y_pow2)
        
        return dist

if __name__ == '__main__':
    points = [[0.2, 0.5],[0.3, 0.4], [0.3, 0.5]]
    dist = get_euclidean_dist(points)
    print dist
    print dist.mean()
    print dist/dist.mean()
    path0 = 'D:\\Downloads\\Python-Shape-Context-master\\Python-Shape-Context-master\\9.png'
    path1 = 'D:\\Downloads\\Python-Shape-Context-master\\Python-Shape-Context-master\\test1.png'
    path2 = 'E:\\workplace\\captcha\\gesture\\1.jpg'    
    path3 = 'E:\\workplace\\captcha\\gesture\\2.jpg'
    path4 = 'E:\\workplace\\captcha\\00.png'
    path5 = 'E:\\workplace\\captcha\\50.png'
    path6 = 'E:\\workplace\\captcha\\40.png'
    p0, T0 = uti.get_points_from_image(path6, simpleto=20)
    p1, T1 = uti.get_points_from_image(path5, simpleto=20)
    shape = sc.ShapeContext()
    
    sc0 = shape.getShapeContext(p0)
    sc1 = shape.getShapeContext(p1)
    diff, indexs = shape.get_diff(sc0, sc1)
    print sc0
    print sc1
    print diff