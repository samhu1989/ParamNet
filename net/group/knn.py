# -*- coding: utf-8 -*-
import tensorflow as tf;
import os;
from tensorflow.python.framework import ops
path = os.path.dirname(os.path.realpath(__file__));
group_module=tf.load_op_library(path+'/group.so');

def knn(xyz,k):
	'''
Computes the distance of k nearest neighbors inside a point clouds
output: dist:  (batch_size,#point,k)   nearest k neighbor dist inside point set 
output: idx:  (batch_size,#point,k)   nearest k neighbor index inside point set
	'''
	return group_module.knn(xyz,k);


if __name__=='__main__':
    import numpy as np;
    import random;
    import time;
    xyz=np.random.randn(32,1024,3).astype('float32');
    with tf.Session('') as sess:
        with tf.device('/cpu:0'):
            ixyzcpu=tf.Variable(xyz)
            distcpu,idxcpu=knn(ixyzcpu,8);
            print idxcpu.shape;
        sess.run(tf.global_variables_initializer())
        t0=time.time();
        for i in xrange(100):
            valcpu = sess.run(idxcpu);
        dvalcpu,valcpu = sess.run([distcpu,idxcpu]);
        cputime = time.time()-t0;
        with tf.device('/gpu:0'):
            ixyzgpu=tf.Variable(xyz)
            distgpu,idxgpu=knn(ixyzgpu,8);
            print idxgpu.shape;
        sess.run(tf.global_variables_initializer())
        t0=time.time();
        for i in xrange(100):
            valgpu = sess.run(idxgpu);
        dvalgpu,valgpu = sess.run([distgpu,idxgpu]);
        gputime = time.time()-t0;
    print "cputime",cputime;
    print "gputime",gputime;
    print "xyz:",xyz.shape;
    #print xyz
    print "valgpu:",valgpu.shape;
    #print valgpu;
    print "valcpu:",valcpu.shape;
    #print valcpu;
    print "(valgpu==valcpu) is ",(valgpu==valcpu).all();
    if not (valgpu==valcpu).all():
        itemindex = np.where( valgpu != valcpu )
        print itemindex;
        print valgpu[itemindex];
        print valcpu[itemindex];
        print dvalcpu[itemindex[:-1]];
        print dvalgpu[itemindex[:-1]];

