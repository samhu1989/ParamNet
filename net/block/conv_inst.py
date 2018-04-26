import tensorflow as tf;
import numpy as np;
import tflearn;

def instance_norm(x, eps=1e-05, decay=0.9, affine=True, name=None):
    with tf.variable_scope(name, default_name='InstatnceNorm2d'):
        params_shape = x.shape[-1:];
        mean, var = tf.nn.moments(x, [a for a in range(1,len(x.shape)-1)] ,keep_dims=True,name='moments');
        if affine:
            beta = tf.get_variable('beta', params_shape,
                                   initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', params_shape,
                                    initializer=tf.ones_initializer)
            x = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, var, None, None, eps)
    return x;

def conv_2d(x,ch,k,strides,activation,weight_decay,regularizer,name):
    x=tflearn.layers.conv.conv_2d(x,ch,k,strides=strides,activation='linear',weight_decay=weight_decay,regularizer=regularizer);
    x=instance_norm(x,name="instn"+name);
    if activation=="relu":
        x=tf.nn.relu(x);
    return x;

def conv_2d_transpose(x,ch,k,sx,strides,activation,weight_decay,regularizer,name):
    x = tflearn.layers.conv.conv_2d_transpose(x,ch,k,sx,strides=strides,activation='linear',weight_decay=weight_decay,regularizer=regularizer);
    x=instance_norm(x,name="instn"+name);
    if activation=="relu":
        x=tf.nn.relu(x);
    return x;

def encoder_instn(x):
#192x256
    x=conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name="00");
    x=conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name="01");
    x0=x
    x=conv_2d(x,32,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2',name='02');
#96x128
    x=conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name='03');
    x=conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name='04');
    x1=x
    x=conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2',name='05');
#48x64
    x=conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name='06');
    x=conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name='07');
    x2=x
    x=conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2',name='08');
#24x32
    x=conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name='09');
    x=conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name='10');
    x3=x
    x=conv_2d(x,256,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2',name='11');
#12x16
    x=conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name='12');
    x=conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name='13');
    x4=x
    x=conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2',name='14');
#6x8
    x=conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name='15');
    x=conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name='16');
    x=conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name='17');
    x5=x
    x=conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2',name='18');
#3x4
    x6=x
    return [x0,x1,x2,x3,x4,x5,x6];

def decoder_instn(xs,name="decoder"):
    x=conv_2d_transpose(xs[6],256,[5,5],[6,8],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2',name=name+"00");
#6 8
    x5=conv_2d(xs[5],256,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2',name=name+"01");
    x=tf.nn.relu(tf.add(x,x5))
    x=conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name=name+"02")
    x=conv_2d_transpose(x,128,[5,5],[12,16],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2',name=name+"03");
#12 16
    x4=conv_2d(xs[4],128,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2',name=name+"04")
    x=tf.nn.relu(tf.add(x,x4))
    x=conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name=name+"05")
    x=conv_2d_transpose(x,64,[5,5],[24,32],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2',name=name+"06")
#24 32
    x3=conv_2d(xs[3],64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2',name=name+"07")
    x=tf.nn.relu(tf.add(x,x3))
    x=conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name=name+"08")
    x=conv_2d_transpose(x,32,[5,5],[48,64],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2',name=name+"09")
#48 64
    x2=conv_2d(xs[2],32,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2',name=name+"10")
    x=tf.nn.relu(tf.add(x,x2))
    x=conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name=name+"11")
    x=conv_2d_transpose(x,16,[5,5],[96,128],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2',name=name+"12")
#96 128
    x1=conv_2d(xs[1],16,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2',name=name+"13");
    x=tf.nn.relu(tf.add(x,x1));
    x=conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name=name+"14");
    x=conv_2d(x,32,(3,3),strides=2,activation='linear',weight_decay=1e-5,regularizer='L2',name=name+"15");
#48 64
    return x;