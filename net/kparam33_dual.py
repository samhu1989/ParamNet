import tensorflow as tf;
import numpy as np;
import loss;
import tflearn;
import block;
import group;
from kp import *;

def KPARAM_33_DUAL(settings={}):
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
        kpN = 4;
        uplevel = 2;
        net['rand'] = "util.rand_sphere_interp(%d,66,%d)"%(BATCH_SIZE,uplevel);
        for i in range(kpN):
            x = block.decoder_instn(xs,name="decoder%02d"%i);
            x3D = kpblock6ext(x,x3D,BATCH_SIZE,k,name="kpblock%d"%i);
            if i < uplevel:
                eidx = tf.placeholder(tf.int32,shape=[None,2],name='eidx_%d'%i);
                net['eidx_%d'%i] = eidx;
                interpx3D = tf.gather(x3D,eidx,axis=1);
                interpx3D = tf.reduce_mean(interpx3D,axis=2);
                x3D = tf.concat([x3D,interpx3D],axis=1);
        x3D = kpblock6dn(x3D,BATCH_SIZE,k,name="kpblockdn");
        x = block.decoder_instn(xs,name="decoderg");
        x3D,scalereg = kpblock6g(x,x3D,BATCH_SIZE);
        net['ox3D'] = x3D;
        #
        dists_forward,_,dists_backward,_= loss.ChamferDistLoss.Loss(yGT,x3D)
        dists_forward=tf.reduce_mean(dists_forward);
        dists_backward=tf.reduce_mean(dists_backward); 
        tf.summary.scalar("dists_forward",dists_forward);
        tf.summary.scalar("dists_backward",dists_backward);
        loss_nodecay=(dists_forward+dists_backward)*1024*100;
        tf.summary.scalar("loss_no_decay",loss_nodecay);
        decay = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))*0.1;
        tf.summary.scalar("decay",decay);
        net['chmf'] = loss_nodecay;
        loss_with_decay = loss_nodecay + decay + scalereg;
        tf.summary.scalar("loss_with_decay",loss_with_decay);
        #
        fidx = tf.placeholder(tf.int32,shape=[None,3],name='fidx');
        net['fidx'] = fidx;
        fx3D = tf.reduce_mean( tf.gather( x3D , fidx, axis=1 ), axis=2, name='fx3D' );
        fdf,_,fdb,_ = loss.ChamferDistLoss.Loss(yGT,fx3D);
        fdf = tf.reduce_mean(fdf);
        fdb = tf.reduce_mean(fdb);
        floss_nodecay = ( fdf + fdb )*1024*100;
        tf.summary.scalar("floss_no_decay",floss_nodecay);
        dual_loss_nodecay = floss_nodecay + loss_nodecay;
        tf.summary.scalar("dual_loss_nodecay",dual_loss_nodecay);
        dual_loss_decay = dual_loss_nodecay + decay + scalereg;
        tf.summary.scalar("dual_loss_decay",dual_loss_decay);
        #
        mse = 2*1024*100*tf.reduce_mean(tf.reduce_sum(tf.square(x3D - yGT),axis=2));
        mse_with_decay = mse + decay + scalereg;
        #
        lr  = tf.placeholder(tf.float32,name='lr');
        net['lr'] = lr;
        stepinit = tf.constant_initializer(0);
        gstep = tf.get_variable(shape=[],initializer=stepinit,trainable=False,name='step',dtype=tf.int32);
        net['step'] = gstep;
        optdchmf = tf.train.AdamOptimizer(lr).minimize(dual_loss_decay,global_step=gstep);
        net['optdchmf'] = optdchmf;
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