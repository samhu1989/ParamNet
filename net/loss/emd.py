import tensorflow as tf;
import os;
import sys;
from scipy.sparse import dok_matrix;
from scipy.sparse import csc_matrix;
import numpy as np;
import time;

def emd_dist(xyz1,xyz2,eps=5e-4,m=1e3):
    n_p = int(xyz1.shape[0]);
    n_o = int(xyz2.shape[0]);
    price = np.zeros(n_o,dtype=np.float32);
    price.fill(m);
    idx_p = np.zeros(n_p,dtype=np.int32);
    idx_p.fill(-1);
    idx_o = np.zeros(n_o,dtype=np.int32);
    idx_o.fill(-1);
    #assigned object number
    a_n_o = 0;
    cost_1st = np.zeros(n_p,dtype=np.float32);
    cost_1st_idx = np.zeros(n_p,dtype=np.int32);
    cost_2nd = np.zeros(n_p,dtype=np.float32);
    cost_2nd_idx = np.zeros(n_p,dtype=np.int32);
    isbid_o = np.zeros(n_o,dtype=np.int32);
    while a_n_o < n_o:
        #biding phase
        isbid_o.fill(0);
        b = dok_matrix((n_p,n_o),dtype=np.float32);
        #reset cost record;
        for i in range(n_p):
            if idx_p[i] == -1:
                c = price - np.sum(np.square(xyz2[:,:] - xyz1[i,:]),axis=1);
                cost_1st_idx[i] = np.argmin(c);
                cost_1st[i] = c[cost_1st_idx[i]];
                c[cost_1st_idx[i]] = 10*m;
                cost_2nd_idx[i] = np.argmin(c);
                cost_2nd[i] = c[cost_2nd_idx[i]];
                if cost_2nd_idx[i] == -1:
                    r = eps;
                else:
                    r = eps + ( cost_2nd[i] - cost_1st[i] );
                b[i,cost_1st_idx[i]] = price[cost_1st_idx[i]] + r;
                isbid_o[cost_1st_idx[i]] = 1;
        #assign
        idx = b.asformat('csc').argmax(axis=0);
        for j in range(n_o):
            if isbid_o[j]==1:
                price[j] = b[idx[0,j],j];
                #the original assignment is released
                if idx_o[j]!=-1:
                    idx_p[idx_o[j]]=-1;
                idx_o[j] = idx[0,j];
                idx_p[idx[0,j]] = j;
        a_n_o = np.sum(idx_o!=-1);
    dist = np.sum(np.square(xyz1 - xyz2[idx_o,:]),axis=1);
    return np.mean(dist),idx_o;
    
def emd_eval(xyz1,xyz2):
    b = xyz1.shape[0];
    n = xyz1.shape[1];
    d = np.zeros(b,dtype=np.float32);
    for bi in range(b):
        d[bi],_ = emd_dist(xyz1[bi,...],xyz2[bi,...]);
    return np.mean(d);