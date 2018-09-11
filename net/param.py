import tensorflow as tf
import tensorflow.contrib.slim as slim;
from .resnet import resnet_v1_18;
from tensorflow.contrib.layers.python.layers import initializers;
from tensorflow.contrib.layers.python.layers import regularizers;
from tensorflow.python.framework import ops;
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from .loss import ChamferDistLoss;
import sys;
from .surf import surf_decoder;

def param_base(settings,surf_name=sys._getframe().f_code.co_name,surf_lst=[['C'],['M',[1024,512]]]):
    if 'batch_size' in settings.keys():
        BATCH_SIZE=settings['batch_size'];
    else:
        BATCH_SIZE=32;
    if 'height' in settings.keys():
        HEIGHT=settings['height'];
    else:
        HEIGHT=192;
    if 'width' in settings.keys():
        WIDTH=settings['width'];
    else:
        WIDTH=256; 
    if 'pts_num' in settings.keys():
        PTS_NUM=settings['pts_num'];
    else:
        PTS_NUM=2500;
    if 'grid_dim' in settings.keys():
        GRID_DIM=settings['grid_dim'];
    else:
        GRID_DIM=3;
    if 'grid_num' in settings.keys():
        GRID_NUM = settings['grid_num'];
    else:
        GRID_NUM = 1;
    if 'dev' in settings.keys():
        dev = settings['dev'];
    else:
        dev = '/gpu:0';
    if 'reuse' in settings.keys():
        reuse = settings['reuse'];
    else:
        reuse = False;
    net = {};
    with tf.device( dev ):
        yGT = tf.placeholder(tf.float32,shape=[None,None,3],name='yGT');
        net['yGT'] = yGT;
        #
        x2D = tf.placeholder(tf.float32,shape=[None,HEIGHT,WIDTH,4],name='x2D');
        net['ix2D'] = x2D;
        #
        xGrid = tf.placeholder(tf.float32,shape=[None,None,GRID_DIM],name='xGrid');
        net['ixGrid'] = xGrid;
        net['rand'] = 'rand_grid(self.BATCH_SIZE,%d,%d)'%(PTS_NUM//GRID_NUM,GRID_DIM);
        #
        isTrain = tf.placeholder(tf.bool,name='isTrain');
        net['isTrain'] = isTrain;
        #
        shape_sig_ext = tf.placeholder(tf.float32,shape=[None,1024],name='shape_sig_ext');
        net['ixShapeSig'] = shape_sig_ext;
        #
        shape_sig , _ = resnet_v1_18(x2D,num_classes=1024,is_training=isTrain,reuse=reuse);
        net['oxShapeSig'] = shape_sig;
        #
        surf_dict = {};
        surf_dict['surf'] = surf_lst;
        surf_dict['grid']='ixGrid';
        #
        surf_dict['sig']='ixShapeSig';
        surf_dict['output']='yext';
        surf_decoder(surf_dict,net,GRID_NUM,isTrain,scope="decoder",reuse=reuse);
        yext = net['yext'];
        #
        surf_dict['sig']='oxShapeSig';
        surf_dict['output']='y';
        surf_decoder(surf_dict,net,GRID_NUM,isTrain,scope="decoder",reuse=True);
        y = net['y'];
        #
        dists_forward,_,dists_backward,_ = ChamferDistLoss.Loss(yGT,y);
        dists_forward=tf.reduce_mean(dists_forward);
        dists_backward=tf.reduce_mean(dists_backward);
        tf.summary.scalar("dists_forward",dists_forward);
        tf.summary.scalar("dists_backward",dists_backward);
        #
        loss_nodecay=( dists_forward + dists_backward )*1024*100;
        tf.summary.scalar("loss_no_decay",loss_nodecay);
        #
        decay = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))*0.1;
        tf.summary.scalar("decay",decay);
        #
        loss_with_decay = loss_nodecay + decay;
        net['chmf'] = loss_with_decay;
        tf.summary.scalar("loss_with_decay",loss_with_decay);
        #
        lr  = tf.placeholder(tf.float32,name='lr');
        net['lr'] = lr;
        stepinit = tf.constant_initializer(0);
        with tf.variable_scope('Common') as scope:
            try:
                gstep = tf.get_variable(shape=[],initializer=stepinit,trainable=False,name='step',dtype=tf.int32);
                optchmf = tf.train.AdamOptimizer(lr).minimize(loss_with_decay,global_step=gstep);
            except:
                scope.reuse_variables();
                gstep = tf.get_variable(name='step',dtype=tf.int32);
                optchmf = tf.train.AdamOptimizer(lr).minimize(loss_with_decay,global_step=gstep);
        net['step'] = gstep;
        net['optchmf'] = optchmf;
        net['sum'] = tf.summary.merge_all();
        if reuse:
            print(surf_name+' rebuilt');
        else:
            print(surf_name+' built');
    return net;

def PARAM001(settings):
    surf=[['C'],['M',[1024,512,256]]];
    return param_base(settings,surf_lst=surf);

def PARAM002(settings):
    surf=[['C'],['M',[1024,512]]];
    return param_base(settings,surf_lst=surf);

def PARAM003(settings):
    surf=[['O'],['M',[1024,512]]];
    return param_base(settings,surf_lst=surf);

def PARAM013(settings):
    surf=[['C'],['P',[4]],['R',[4,64]]];
    return param_base(settings,surf_lst=surf);

def PARAM014(settings):
    surf=[['C'],['P',[8]],['R',[8,64]]];
    return param_base(settings,surf_lst=surf);

def PARAM015(settings):
    surf=[['C'],['P',[16]],['R',[16,64]]];
    return param_base(settings,surf_lst=surf);

def PARAM016(settings):
    surf=[['O'],['P',[4]],['R',[4,64]]];
    return param_base(settings,surf_lst=surf);

def PARAM017(settings):
    surf=[['O'],['P',[8]],['R',[8,64]]];
    return param_base(settings,surf_lst=surf);

def PARAM018(settings):
    surf=[['O'],['P',[16]],['R',[16,64]]];
    return param_base(settings,surf_lst=surf);
    