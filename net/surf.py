from __future__ import absolute_import;
from __future__ import division;
from __future__ import print_function;

import tensorflow as tf;
import tensorflow.contrib.slim as slim;
from tensorflow.contrib.layers.python.layers import initializers;
from tensorflow.contrib.layers.python.layers import regularizers;
from tensorflow.contrib import layers;
from tensorflow.contrib.framework.python.ops import add_arg_scope;
from tensorflow.contrib.framework.python.ops import arg_scope;
from tensorflow.contrib.layers.python.layers import layers as layers_lib;
from tensorflow.contrib.layers.python.layers import utils;
from tensorflow.contrib.slim.python.slim.nets import resnet_utils;
from tensorflow.python.ops import math_ops;
from tensorflow.python.ops import nn_ops;
from tensorflow.python.ops import variable_scope;
from tensorflow.python.framework import ops;
from .group import knn;

resnet_arg_scope = resnet_utils.resnet_arg_scope;

def kconv(x,k,knn_index,d,scope,is_training,reuse):
    weight_decay=1e-4;
    batch_norm_decay=0.997;
    batch_norm_epsilon=1e-5;
    batch_norm_scale=True;
    batch_norm_params={'decay':batch_norm_decay,'epsilon':batch_norm_epsilon,'scale':batch_norm_scale,'is_training':is_training,'updates_collections': ops.GraphKeys.UPDATE_OPS}
    with arg_scope([slim.conv2d],
      weights_regularizer=regularizers.l2_regularizer(weight_decay),
      weights_initializer=initializers.variance_scaling_initializer(),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params,
      reuse=reuse):
        with arg_scope([slim.batch_norm], **batch_norm_params):
            if int(x.shape[-1]) > 256:# too large to fit in gpu
                x  = tf.reshape(x,[tf.shape(x)[0],tf.shape(x)[1],1,int(x.shape[2])]);
                x = slim.conv2d(x,256,[1,1],scope=scope+'_reducedim');
            rx = tf.reshape(x,[tf.shape(x)[0],tf.shape(x)[1],int(x.shape[-1])]);
            xknn = tf.gather_nd(rx,knn_index,name=scope+'_gather');
            x = tf.concat([tf.reshape(x,[tf.shape(x)[0],tf.shape(x)[1],1,int(x.shape[-1])]),xknn],axis=2);
            x = slim.conv2d(x,d,[1, k+1],scope=scope+"_kconv",padding='VALID');
    return tf.reshape(x,[tf.shape(x)[0],-1,d]);

def duplicate_concate(sig,grid,y=None):
    x = tf.reshape(sig,[tf.shape(sig)[0],1,sig.shape[-1]]);
    x = tf.tile(x,[1,tf.shape(y)[1],1]);
    x = tf.concat([y,x],axis=2);
    return x;

def duplicate_outerproduct(sig,grid,y=None):
    x = tf.reshape(sig,[tf.shape(sig)[0],1,sig.shape[-1],1]);
    x = tf.tile(x,[1,tf.shape(grid)[1],1,1]);
    y = tf.reshape(grid,[tf.shape(grid)[0],tf.shape(grid)[1],1,int(grid.shape[-1])]);
    return tf.reshape(tf.matmul(x,y),[tf.shape(grid)[0],tf.shape(grid)[1],int(sig.shape[-1])*int(grid.shape[-1])]);

def kconv_couple(sig,grid,y=None,param=None):
    return None;

def mlp(sig,grid,siggrid,param=None):
    scope = param[0]
    grid_num = param[1];
    is_training = param[2];
    reuse = param[3];
    weight_decay=1e-4;
    batch_norm_decay=0.997;
    batch_norm_epsilon=1e-5;
    batch_norm_scale=True;
    batch_norm_params={'decay':batch_norm_decay,'epsilon':batch_norm_epsilon,'scale': batch_norm_scale,'is_training':is_training,'updates_collections': ops.GraphKeys.UPDATE_OPS}
    ix =  tf.reshape(siggrid,[tf.shape(siggrid)[0],1,tf.shape(siggrid)[1],siggrid.shape[-1]]);
    oy = [];
    with arg_scope([slim.conv2d],
      weights_regularizer=regularizers.l2_regularizer(weight_decay),
      weights_initializer=initializers.variance_scaling_initializer(),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params,
      reuse=reuse):
        with arg_scope([slim.batch_norm], **batch_norm_params):
            for i in range(grid_num):
                x = ix;
                for j,size in enumerate(param[-1]): 
                    x = slim.conv2d(x, size, [1, 1],scope=scope+"_grid%d_fc%d"%(i,j));
                oy.append( slim.conv2d(x, 3, [1, 1] ,activation_fn = tf.nn.tanh,scope=scope+"_grid%d_fc%d"%(i,j+1)) );
            y = tf.concat(oy,axis=2);
            return tf.reshape(y,[tf.shape(siggrid)[0],-1,3]);

def kconv_pts(sig,grid,siggrid,param=None):
    scope = param[0];
    grid_num = param[1];
    is_training = param[2];
    reuse = param[3];
    kidx = param[4];
    k = param[-1][0];
    if not kidx:
        _,knn_index = knn(grid,k);
        kidx.append(knn_index);
    else:
        knn_index = kidx[0];
    return kconv(siggrid,k,knn_index,3,scope,is_training,reuse);

def kconv_pts_res(sig,grid,y,param=None):
    scope = param[0];
    gridn = param[1];
    is_training = param[2];
    reuse = param[3];
    kidx = param[4];
    k = param[-1][0];
    d = param[-1][1];
    if not kidx:
        _,knn_index = knn(grid,k);
        print(knn_index.shape);
        kidx.append(knn_index);
    else:
        knn_index = kidx[0];
    short = y;
    res = y;
    res = kconv(res,k,knn_index,d,scope+'_1',is_training,reuse);
    res = tf.nn.relu(res);
    res = kconv(res,k,knn_index,3,scope+'_2',is_training,reuse);
    return tf.nn.relu(short+res);

def laplace(sig,grid,y,param=None):
    lplidx = param[-1][0];
    lplw = param[-1][1];
    lpln = tf.gather(y,lplidx,axis=1);
    return tf.reduce_sum(lpln*tf.reshape(lplw,[1,-1,6,1]),axis=2);

blocks={
    'M':mlp,
    'C':duplicate_concate,
    'O':duplicate_outerproduct,
    'K':kconv_couple,
    'P':kconv_pts,
    'R':kconv_pts_res,
    'L':laplace
};

def surf_decoder(surf_dict,net,grid_num,istrain,scope='surf',reuse=False):
    if net:
        sig = net[surf_dict['sig']];
        grid = net[surf_dict['grid']];
        surf = surf_dict['surf'];
        kidx = [];
        y = grid;
        for i,b in enumerate(surf):
            with tf.variable_scope(scope+'_'+b[0]+"%02d"%i,reuse=reuse):
                if len(b) == 2:
                    args = [scope+'_'+b[0]+"%02d"%i,grid_num,istrain,reuse,kidx,b[1]];
                    y = blocks[b[0]](sig,grid,y,param=args);
                if len(b) == 1:
                    y = blocks[b[0]](sig,grid,y);
        net[surf_dict['output']] = y;
    