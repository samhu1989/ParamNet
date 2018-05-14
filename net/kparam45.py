import tensorflow as tf;
import numpy as np;
import loss;
import tflearn;
import block;
import group;
from kp import *;

def KPARAM_45(settings={}):
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
    PTS_DIM=3;
    if 'dev' in settings.keys():
        dev = settings['dev'];
    else:
        dev = '/gpu:0';
    net = {};
    with tf.device( dev ):
        tflearn.init_graph(seed=1029,num_cores=1,gpu_memory_fraction=0.9,soft_placement=True);
        yGT = tf.placeholder(tf.float32,shape=[None,PTS_DIM],name='yGT');
        net['yGT'] = yGT;
        yGT = tf.reshape(yGT,[BATCH_SIZE,-1,PTS_DIM]);
        x2D = tf.placeholder(tf.float32,shape=[None,HEIGHT,WIDTH,4],name='x2D');
        net['ix2D'] = x2D;
        x2D = tf.reshape(x2D,[BATCH_SIZE,HEIGHT,WIDTH,4]);
        x3D = tf.placeholder(tf.float32,shape=[None,PTS_DIM],name='x3D');
        net['ix3D'] = x3D;
        x3D = tf.reshape(x3D,[BATCH_SIZE,-1,PTS_DIM]);
        x = x2D;
        xs = block.encoder_instn(x);
        k = 16;
        kpN = 3;
        net['rand'] = "util.rand_sphere_nointerp(%d,1024)"%(BATCH_SIZE);
        lplidx = tf.placeholder(tf.int32,shape=[None,6],name='lplidx');
        net['lplidx_0'] = lplidx;
        lplw = tf.placeholder(tf.float32,shape=[None,6],name='lplw');
        net['lplw_0'] = lplw;
        for i in range(kpN):
            x = block.decoder_instn(xs,name="decoder%02d"%i);
            x3D = kpblock6ext(x,x3D,BATCH_SIZE,k,name="kpblock%d"%i);
            x3D = lplsmooth2(x3D,lplidx,lplw,name='kpsmooth%d'%i);
        #x3D = kpblock6dn(x3D,BATCH_SIZE,k,name="kpblockdn");
        x = block.decoder_instn(xs,name="decoderg");
        x3D,scalereg = kpblock6g(x,x3D,BATCH_SIZE);
        net['ox3D'] = x3D;
        eidx = tf.placeholder(tf.int32,shape=[None,2],name='eidx_0');
        net['eidx_0'] = eidx;
        x3Dedge = tf.gather(x3D,eidx,axis=1);
        x3Dedgedif = x3Dedge*tf.constant([1,1,1,-1,-1,-1],dtype=tf.float32,shape=[1,1,2,3],name='substract');
        x3Dedgevec = tf.reduce_sum(x3Dedgedif,axis=2,keep_dims=True);
        x3Dedgelen2 = tf.sqrt(tf.reduce_sum(tf.square(x3Dedgevec),axis=[2,3]));
        x3Dedgevar = tf.reduce_sum(tf.square(x3Dedgelen2 - tf.reduce_mean(x3Dedgelen2,axis=1,keep_dims=True)));
        tf.summary.scalar("x3Dedgevar",x3Dedgevar);
        #
        dists_forward,_,dists_backward,idxb = loss.ChamferDistLoss.Loss(yGT,x3D);
        dists_forward=tf.reduce_mean(dists_forward);
        dists_backward=tf.reduce_mean(dists_backward);
        tf.summary.scalar("dists_forward",dists_forward);
        tf.summary.scalar("dists_backward",dists_backward);
        #
        ynGT = tf.placeholder(tf.float32,shape=[BATCH_SIZE,None,PTS_DIM],name='ynGT');
        net['ynGT'] = ynGT;
        idxnearest_constv = np.ones([BATCH_SIZE,1024],dtype=np.int32)*(np.array([x for x in range(BATCH_SIZE)],dtype=np.int32).reshape((BATCH_SIZE,1)));
        idxnearest_const = tf.constant(idxnearest_constv,shape=[BATCH_SIZE,1024,1],dtype=tf.int32,name='idxnearest_const');
        idxnearest = tf.concat([idxnearest_const,tf.reshape(idxb,[BATCH_SIZE,1024,1])],2);
        x3Dn = tf.gather_nd(ynGT,idxnearest);
        x3Dnedge = tf.gather(x3D,eidx,axis=1);
        norm_loss = tf.reduce_sum(tf.square(tf.reduce_sum(x3Dnedge*x3Dedgevec,axis=3)));
        tf.summary.scalar("norm_loss",norm_loss);
        #
        loss_nodecay=(dists_forward+dists_backward)*1024*100;
        tf.summary.scalar("loss_no_decay",loss_nodecay);
        decay = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))*0.1;
        tf.summary.scalar("decay",decay);
        net['chmf'] = loss_nodecay;
        loss_with_decay = loss_nodecay + decay + scalereg + x3Dedgevar;
        tf.summary.scalar("loss_with_decay",loss_with_decay);
        #
        mse = 2*1024*100*tf.reduce_mean(tf.reduce_sum(tf.square(x3D - yGT),axis=2));
        mse_with_decay = mse + decay + scalereg;
        #
        lr  = tf.placeholder(tf.float32,name='lr');
        net['lr'] = lr;
        stepinit = tf.constant_initializer(0);
        gstep = tf.get_variable(shape=[],initializer=stepinit,trainable=False,name='step',dtype=tf.int32);
        net['step'] = gstep;
        optchmf = tf.train.AdamOptimizer(lr).minimize(loss_with_decay,global_step=gstep);
        net['optchmf'] = optchmf;
        #
        net['sum'] = tf.summary.merge_all();
        prestepinit = tf.constant_initializer(0);
        prestep = tf.get_variable(shape=[],initializer=prestepinit,trainable=False,name='prestep',dtype=tf.int32);
        net['prestep'] = prestep;
        optmse = tf.train.AdamOptimizer(lr).minimize(mse_with_decay,global_step=prestep);
        net['optmse'] = optmse;
        net['presum'] = tf.summary.merge([tf.summary.scalar("mse",mse),tf.summary.scalar("mse_with_decay",mse_with_decay)]);
        #
    return net;