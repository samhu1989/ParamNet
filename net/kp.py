import tensorflow as tf;
import numpy as np;
import tflearn;
import group;
from block import *;

def kpsmooth(x3D,batch_size,k,name="kpsmooth"):
    _,knn_index = group.knn(x3D,k);
    x3Dkcnnglobal = tf.gather_nd( x3D , knn_index , name = name+"_gather" );
    x3Dkcnnglobal = tf.concat([x3Dkcnnglobal,tf.reshape(x3D,[batch_size,-1,1,3])],axis=2);
    return tf.reduce_mean(x3Dkcnnglobal,axis=2);
    
def lplsmooth(x3D,name='lplsmooth'):
    lplidx = tf.placeholder(tf.int32,shape=[None,6],name=name+'_lplidx');
    lplw = tf.placeholder(tf.float32,shape=[None,6],name=name+'_lplw');
    lpln = tf.gather(x3D,lplidx,axis=1);
    return tf.reduce_sum(lpln*tf.reshape(lplw,[1,-1,6,1]),axis=2),lplidx,lplw;
    
def lplsmooth2(x3D,lplidx,lplw,name='lplsmooth'):
    lpln = tf.gather(x3D,lplidx,axis=1);
    return tf.reduce_sum(lpln*tf.reshape(lplw,[1,-1,6,1]),axis=2);
    
def kpblock6ext(x2D,x3D,batch_size,k,name="kpblock6",reuse=False):
    x2D = conv_2d(x2D,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name=name+"00",reuse=reuse);
    x2D = conv_2d(x2D,8,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name=name+"01",reuse=reuse);
    x2Da = conv_2d(x2D,1,(1,1),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2',name=name+"02",reuse=reuse);
    x2Da = tf.nn.max_pool(x2Da,[1,2,2,1],[1,2,2,1],"VALID");
    x2Db = tf.nn.max_pool(x2D,[1,2,2,1],[1,2,2,1],"VALID");
    x2Db = conv_2d(x2Db,1,(1,1),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2',name=name+"03",reuse=reuse);
    x2D = tf.nn.relu( x2Da + x2Db );
    x2Ddim = int(x2D.shape[1])*int(x2D.shape[2])*int(x2D.shape[3]);
    num = 4*k;
    n = [
        6*(num//2),
        (num//2)*(num//4),
        (num//4)*(num//8),
        (num//8),
        (num//8)*3,
        3
    ]; 
    shape = [ 
        [batch_size,6,num//2],
        [batch_size,num//2,num//4],
        [batch_size,num//4,num//8],
        [batch_size,1,num//8],
        [batch_size,num//8,3],
        [batch_size,1,3]
    ];
    param = [];
    fc_scope = tf.variable_scope(name,reuse=reuse);
    for i in range(len(n)):
        if 2*n[i] < x2Ddim:
            x1D = tflearn.layers.core.fully_connected( x2D,2*n[i],activation='relu',weight_decay=1e-4,regularizer='L2',reuse=reuse,scope=fc_scope);
        else:
            x1D = tflearn.layers.core.fully_connected( x2D,n[i],activation='relu',weight_decay=1e-4,regularizer='L2',reuse=reuse,scope=fc_scope);
        x1D = tflearn.layers.core.fully_connected( x1D,n[i],activation='linear',weight_decay=1e-4,regularizer='L2',reuse=reuse,scope=fc_scope);
        param.append(tf.reshape(x1D,shape[i]));
    _,knn_index = group.knn(x3D,k);
    x3Dkcnnglobal = tf.gather_nd( x3D , knn_index , name = name+"_gather" );
    x3Dkcnnlocal = x3Dkcnnglobal - tf.reshape(x3D,[batch_size,-1,1,3]);
    x3Dkcnn = tf.concat([x3Dkcnnglobal,x3Dkcnnlocal],3);
    x3Dkcnn = tf.reshape(x3Dkcnn,[batch_size,-1,6]);
    x3Dkcnn = tf.matmul(x3Dkcnn,param[0]);
    x3Dkcnn = tf.nn.relu(x3Dkcnn);
    x3Dkcnn = tf.matmul(x3Dkcnn,param[1]);
    x3Dkcnn = tf.reshape(x3Dkcnn,[batch_size,-1,k,num//4]);
    x3Dkcnn = tf.reduce_max(x3Dkcnn,2);
    x3Dkcnn = tf.matmul(x3Dkcnn,param[2]) + param[3];
    x3Dkcnn = tf.nn.relu(x3Dkcnn);
    x3Dkcnn = tf.matmul(x3Dkcnn,param[4]) + param[5];
    x3D += x3Dkcnn;                             
    return x3D;
    
def kpblock6ext_elu(x2D,x3D,batch_size,k,name="kpblock6"):
    x2D = conv_2d(x2D,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name=name+"00");
    x2D = conv_2d(x2D,8,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name=name+"01");
    x2Da = conv_2d(x2D,1,(1,1),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2',name=name+"02");
    x2Da = tf.nn.max_pool(x2Da,[1,2,2,1],[1,2,2,1],"VALID");
    x2Db = tf.nn.max_pool(x2D,[1,2,2,1],[1,2,2,1],"VALID");
    x2Db = conv_2d(x2Db,1,(1,1),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2',name=name+"03");
    x2D = tf.nn.relu( x2Da + x2Db );
    x2Ddim = int(x2D.shape[1])*int(x2D.shape[2])*int(x2D.shape[3]);
    num = 4*k;
    n = [
        6*(num//2),
        (num//2)*(num//4),
        (num//4)*(num//8),
        (num//8),
        (num//8)*3,
        3
    ]; 
    shape = [ 
        [batch_size,6,num//2],
        [batch_size,num//2,num//4],
        [batch_size,num//4,num//8],
        [batch_size,1,num//8],
        [batch_size,num//8,3],
        [batch_size,1,3]
    ];
    param = [];
    for i in range(len(n)):
        if 2*n[i] < x2Ddim:
            x1D = tflearn.layers.core.fully_connected( x2D,2*n[i],activation='relu',weight_decay=1e-4,regularizer='L2');
        else:
            x1D = tflearn.layers.core.fully_connected( x2D,n[i],activation='relu',weight_decay=1e-4,regularizer='L2');
        x1D = tflearn.layers.core.fully_connected( x1D,n[i],activation='linear',weight_decay=1e-4,regularizer='L2');
        param.append(tf.reshape(x1D,shape[i]));
    _,knn_index = group.knn(x3D,k);
    x3Dkcnnglobal = tf.gather_nd( x3D , knn_index , name = name+"_gather" );
    x3Dkcnnlocal = x3Dkcnnglobal - tf.reshape(x3D,[batch_size,-1,1,3]);
    x3Dkcnn = tf.concat([x3Dkcnnglobal,x3Dkcnnlocal],3);
    x3Dkcnn = tf.reshape(x3Dkcnn,[batch_size,-1,6]);
    x3Dkcnn = tf.matmul(x3Dkcnn,param[0]);
    x3Dkcnn = tf.nn.elu(x3Dkcnn);
    x3Dkcnn = tf.matmul(x3Dkcnn,param[1]);
    x3Dkcnn = tf.reshape(x3Dkcnn,[batch_size,-1,k,num//4]);
    x3Dkcnn = tf.reduce_max(x3Dkcnn,2);
    x3Dkcnn = tf.matmul(x3Dkcnn,param[2]) + param[3];
    x3Dkcnn = tf.nn.elu(x3Dkcnn);
    x3Dkcnn = tf.matmul(x3Dkcnn,param[4]) + param[5];
    x3D += x3Dkcnn;                             
    return x3D;

def kpblock6dn(x3D,batch_size,k,name="kpblock6dn"):
    num = 4*k;
    shape = [ 
        [6,num//2],
        [num//2,num//4],
        [num//4,num//8],
        [1,num//8],
        [num//8,3],
        [1,3]
    ];
    param = [];
    init = tf.glorot_normal_initializer();
    reg = tf.contrib.layers.l2_regularizer(1e-4);
    for i in range(len(shape)):
        param.append(tf.get_variable(shape=shape[i],regularizer=reg,name=name+'param_%d'%i));
    _,knn_index = group.knn(x3D,k);
    x3Dkcnnglobal = tf.gather_nd( x3D , knn_index , name = name+"_gather" );
    x3Dkcnnlocal = x3Dkcnnglobal - tf.reshape(x3D,[batch_size,-1,1,3]);
    x3Dkcnn = tf.concat([x3Dkcnnglobal,x3Dkcnnlocal],3);
    x3Dkcnn = tf.reshape(x3Dkcnn,[-1,6]);
    x3Dkcnn = tf.matmul(x3Dkcnn,param[0]);
    x3Dkcnn = tf.nn.relu(x3Dkcnn);
    x3Dkcnn = tf.matmul(x3Dkcnn,param[1]);
    x3Dkcnn = tf.reshape(x3Dkcnn,[-1,k,num//4]);
    x3Dkcnn = tf.reduce_max(x3Dkcnn,1);
    x3Dkcnn = tf.matmul(x3Dkcnn,param[2]) + param[3];
    x3Dkcnn = tf.nn.relu(x3Dkcnn);
    x3Dkcnn = tf.matmul(x3Dkcnn,param[4]) + param[5];
    x3Dkcnn = tf.reshape(x3Dkcnn,[batch_size,-1,3]);
    x3D += x3Dkcnn;                             
    return x3D;
    
def kpblock6dn_elu(x3D,batch_size,k,name="kpblock6dn"):
    num = 4*k;
    shape = [ 
        [6,num//2],
        [num//2,num//4],
        [num//4,num//8],
        [1,num//8],
        [num//8,3],
        [1,3]
    ];
    param = [];
    init = tf.glorot_normal_initializer();
    reg = tf.contrib.layers.l2_regularizer(1e-4);
    for i in range(len(shape)):
        param.append(tf.get_variable(shape=shape[i],regularizer=reg,name=name+'param_%d'%i));
    _,knn_index = group.knn(x3D,k);
    x3Dkcnnglobal = tf.gather_nd( x3D , knn_index , name = name+"_gather" );
    x3Dkcnnlocal = x3Dkcnnglobal - tf.reshape(x3D,[batch_size,-1,1,3]);
    x3Dkcnn = tf.concat([x3Dkcnnglobal,x3Dkcnnlocal],3);
    x3Dkcnn = tf.reshape(x3Dkcnn,[-1,6]);
    x3Dkcnn = tf.matmul(x3Dkcnn,param[0]);
    x3Dkcnn = tf.nn.elu(x3Dkcnn);
    x3Dkcnn = tf.matmul(x3Dkcnn,param[1]);
    x3Dkcnn = tf.reshape(x3Dkcnn,[-1,k,num//4]);
    x3Dkcnn = tf.reduce_max(x3Dkcnn,1);
    x3Dkcnn = tf.matmul(x3Dkcnn,param[2]) + param[3];
    x3Dkcnn = tf.nn.elu(x3Dkcnn);
    x3Dkcnn = tf.matmul(x3Dkcnn,param[4]) + param[5];
    x3Dkcnn = tf.reshape(x3Dkcnn,[batch_size,-1,3]);
    x3D += x3Dkcnn;                             
    return x3D;

def kpblock6g(x2D,x3D,batch_size,name="kpblock6g",reuse=False):
    x2D = conv_2d(x2D,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name=name+"00",reuse=reuse);
    x2D = conv_2d(x2D,8,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name=name+"01",reuse=reuse);
    x2Da = conv_2d(x2D,1,(1,1),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2',name=name+"02",reuse=reuse);
    x2Da = tf.nn.max_pool(x2Da,[1,2,2,1],[1,2,2,1],"VALID");
    x2Db = tf.nn.max_pool(x2D,[1,2,2,1],[1,2,2,1],"VALID");
    x2Db = conv_2d(x2Db,1,(1,1),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2',name=name+"03",reuse=reuse);
    x2D = tf.nn.relu( x2Da + x2Db );
    x2Ddim = int(x2D.shape[1])*int(x2D.shape[2])*int(x2D.shape[3]);
    n = [
        3
    ]; 
    shape = [ 
        [batch_size,1,3]
    ];
    param = [];
    fc_scope = tf.variable_scope(name,reuse=reuse);
    for i in range(len(n)):
        if 2*n[i] < x2Ddim:
            x1D = tflearn.layers.core.fully_connected( x2D,2*n[i],activation='relu',weight_decay=1e-4,regularizer='L2',reuse=reuse,scope=fc_scope);
        else:
            x1D = tflearn.layers.core.fully_connected( x2D,n[i],activation='relu',weight_decay=1e-4,regularizer='L2',reuse=reuse,scope=fc_scope);
        x1D = tflearn.layers.core.fully_connected( x1D,n[i],activation='linear',weight_decay=1e-4,regularizer='L2',reuse=reuse,scope=fc_scope);
        param.append(tf.reshape(x1D,shape[i]));                          
    return x3D*(tf.nn.sigmoid(param[0]) + 0.3),tf.reduce_mean(tf.square(tf.nn.sigmoid(param[0]) - 0.7));