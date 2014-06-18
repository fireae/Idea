# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 15:54:08 2014

@author: Administrator
"""
import numpy as np
import munkres 
import heapq
import scipy.interpolate as scipolate

def dist2(x,c):
    """
        Euclidian distance matrix
    """
    ncentres = c.shape[0]
    ndata = x.shape[0]
    return (np.ones((ncentres, 1)) * (((np.power(x,2)).H)).sum(axis=0)).H + np.ones((ndata, 1)) * ((np.power(c,2)).H).sum(axis=0) - np.multiply(2,(x*(c.H)));
    
    
def bookenstain(X,Y,beta):
    """
        Bookstein PAMI89
    
        Article: Principal Warps: Thin-Plate Splines and the Decomposition of Deformations

    """
    X = np.asmatrix(X)
    Y = np.asmatrix(Y)

    N = X.shape[0]
    r2 = dist2(X,X)
    K = np.multiply(r2, np.log(r2+ np.eye(N,N)))
    P = np.concatenate(( np.ones((N,1)),X),1)
    L = np.bmat([[K, P], [P.H, np.zeros((3,3))]])
    V = np.concatenate((Y.H, np.zeros((2,3))),1)

    L[0:N,0:N] = L[0:N,0:N] + beta * np.eye(N,N)

    invL = np.linalg.inv(L)

    # L^-1 * v^T = (W | a_1 a_x a_y)^T
    c = invL*(V.H)
    cx = c[:,0]
    cy = c[:,1]
    
    Q = (c[0:N,:].H) * K * c[0:N,:]
    E = np.mean(np.diag(Q))

    n_good = 10

    A = np.concatenate((cx[n_good+2:n_good+3,:],cy[n_good+2:n_good+3,:]),1);
    s = np.linalg.svd(A);
    aff_cost = np.log(s[0]/s[1])

    return cx,cy,E,aff_cost,L

class ShapeContext(object):
    '''
    Shape Context            
    '''
    HUNGURIAN = 1
    def __init__(self, nbins_r=5, nbins_theta=12, 
                 inner_r=0.125, outer_r=2.0):
        self.nbins_r = nbins_r
        self.nbins_theta = nbins_theta
        self.inner_r = inner_r
        self.outer_r = outer_r
        self.nbins = nbins_r*nbins_theta
        
        
    def _get_euclidean_dist(self, points):
        '''
        Parameters:
        @input: points --- list, which elment is like [0.2, 0.5] 
                example: [[0.2, 0.5], [0.3, 0.4]]        
        @output: dist --- np.ndarray
                 examples:   [[ 0.          0.14142136]
                              [ 0.14142136  0.        ]]
        
        @brief:
        1.Coordinates on shape:
        (1) 0.2, 0.5
        (2) 0.3, 0.4
        
        2.Compute eculidean distance from each point to all others:
        [[ 0.          0.14142136]
         [ 0.14142136  0.        ]]
        
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
        
        
    def _get_angles(self, points):
        '''
        @brief: compute angles between all points        
         Parameters:
        @input: points --- list, which elment is like [0.2, 0.5] 
                example: [[0.2, 0.5], [0.3, 0.4]]        
        @output: dist --- np.ndarray
                 examples:   [[ 0.      0.]
                              [ 3.1416  0.]]
                              
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
    
        angles = np.arctan2((p_y_tile - p_y_tile_t), (p_x_tile - p_x_tile_t))        
        return angles
        
        
    def getShapeContext(self, points):
        '''
        step1: compute Euclidean distance from each point to all others:
        step2: normalize by mean distance
        step3: create log distance scale for normalized distances 
        step4: create distance histgoram: iterate for each scale incrementing
                bins when dist < 
        step5: create angles between all ponits(0 to 2*pi)
        step6: Simple Quantization for angles
                theta_array_q = 1 + floor(theta_array_2/(2*pi/nbins_theta))
        step7: combine into a descriptor
        '''
        # step1. compute euclidean distance from each point to all others
        euclidean_dist = self._get_euclidean_dist(points)
        
        # step2. normalize by mean distance
        d_mean = euclidean_dist.mean()
        dist_mean = euclidean_dist / d_mean
        
        # step3. create log distance scale
        log_dist_scale = np.logspace(np.log10(self.inner_r), 
                                     np.log10(self.outer_r), self.nbins_r)
                                     
        # step4. create distance histgoram                                   
        row = len(points)
        dist_hist = np.zeros((row, row), dtype=np.int)
        for i in range(self.nbins_r):
            dist_hist += (dist_mean < log_dist_scale[i])
        fz = dist_hist > 0
        
        # step5: create angles between all ponits(0 to 2*pi)
        angles = self._get_angles(points)
        angles_2 = angles + 2*np.pi* (angles<0)
        angles_theta = np.zeros((row, row), dtype=np.int)
        angles_theta = 1+np.floor(angles_2/(2.0*np.pi / self.nbins_theta))
        
        # step6: combine to one descriptor
        SC = np.zeros((row, self.nbins))
        for i in xrange(row):
            sn = np.zeros((self.nbins_r, self.nbins_theta), dtype=np.int)
            for j in xrange(row):
                if (fz[i,j]):
                    sn[dist_hist[i,j] -1, angles_theta[i,j]-1] += 1
            SC[i] = sn.reshape(self.nbins)
        return SC
        
    def _get_point_cost(self, hi, hj):
        '''
        Let C_ij = C(p_i, q_j) denote the cost of matching these two points.
        C_ij = 1/2 * \sum_{k=1}^{K} \frac{[h_i(k) - h_j(k)]^2}{h_i(k)+h_j(k)}
        '''
        c = 0
        for k in range(min(len(hi), len(hj))):
            if (hi[k] + hj[k]): 
                c = c + (hi[k] - hj[k]) ** 2 / (hi[k] + hj[k])
        C_ij = 0.5 * c
        return C_ij
        
    def get_cost(self, P, Q, qlength=None):
        '''
            Shape Context P and Q,
            compute the cost of points between each other
        '''
        p_row, _ = P.shape
        q_row, _ = P.shape
        C = np.zeros((p_row, q_row), dtype=np.float64)
        
        d = q_row
        if qlength:
            d = qlength
        for i in xrange(p_row):
            for j in xrange(q_row):
                C[i, j] = self._get_point_cost(Q[j]/d, P[i]/p_row)
        return C
        
    def _hungurian_method(self, C):
        m = munkres.Munkres()
        indexes = m.compute(C.tolist())
        total = 0
        for row, col in indexes:
            value = C[row][col]
            total = total + value
            
        return total, indexes
        
    def get_diff(self, P, Q, qlength=None, method=HUNGURIAN):
        """
            if Q is generalized shape context then it compute shape match.
            
            if Q is r point representative shape contexts and qlength set to 
            the number of points in Q then it compute fast shape match.
                
        """
        result = None
        C = self.get_cost(P, Q, qlength)
        
        if method == self.HUNGURIAN:
            result = self._hungurian_method(C)
        else:
            raise Exception('No such optimization')
            
        return result
        
    def get_contexts(self, BH, r=5):
        '''
        # get r shape contexts with maximum number(-BH[i]) or minimum(+BH[i]) of connected elements
        # this gives same result for same query        
        '''        
        res = np.zeros((r, self.nbins))
        used = []
        sums = []
        
        for i in xrange(len(BH)):
            heapq.heappush(sums, (BH[i].sum(), i))
        
        for i in xrange(r):
            _, l = heapq.heappop(sums)
            res[i] = BH[l]
            used.append(l)
        
        del sums
        
        return res, used
        
    def interpolate(self, P1, P2):
        assert len(P1) == len(P2), 'Shapes has different number of points'
        x = [0] * len(P1)
        xs = [0] * len(P1)
        y = [0] * len(P1)
        ys = [0] * len(P1)
        for i in xrange(len(P1)):
            x[i] = P1[i][0]
            xs[i] = P2[i][0]
            y[i] = P1[i][1]
            ys[i] = P2[i][1]
        def U(r):
            res = r**2 * np.log(r**2)
            res[ r==0 ] = 0
            return res
        
        SM = 0.1
        
        fx = scipolate.Rbf(x, xs, function=U, smooth=SM)
        fy = scipolate.Rbf(y, ys, function=U, smooth=SM)
        
        cx, cy, E, affcost, L = bookenstain(P1, P2, 15)
        
        return fx, fy, E, float(affcost)
        
if __name__ == '__main__':
    points = [[0.2, 0.5], [0.4, 0.5], [0.3, 0.4], [0.15, 0.3], [0.3, 0.2], [0.45, 0.3]]
    shapecontext = ShapeContext()
    sc = shapecontext.getShapeContext(points)
    print sc
        