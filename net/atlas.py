import tensorflow as tf
import tensorflow.contrib.slim as slim;
from .resnet import resnet_v1_18;
from tensorflow.contrib.layers.python.layers import initializers;
from tensorflow.contrib.layers.python.layers import regularizers;

def atlas_decode(siggrid,gridn,is_training,scope="",reuse=False,bottleneck_size=1024,weight_decay=1e-4,batch_norm_decay=0.997,batch_norm_epsilon=1e-5,batch_norm_scale=True):
    batch_norm_params={'decay': batch_norm_decay,'epsilon':batch_norm_epsilon,'scale': batch_norm_scale,'updates_collections': ops.GraphKeys.UPDATE_OPS}
    ix =  tf.reshape(siggrid,[tf.shape(siggrid)[0],1,tf.shape(siggrid)[1],tf.shape(siggrid)[2]]);
    oy = [];
    with arg_scope([slim.conv2d],
      weights_regularizer=regularizers.l2_regularizer(weight_decay),
      weights_initializer=initializers.variance_scaling_initializer(),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params,
      reuse=reuse):
        with arg_scope([slim.batch_norm], **batch_norm_params):
            for i in range(gridn):
                x = slim.conv2d(ix, bottleneck_size, [1, 1],scope=scope+"_grid%d_fc1"%i);
                x = slim.conv2d(x, bottleneck_size//2, [1, 1],scope=scope+"_grid%d_fc2"%i);
                x = slim.conv2d(x, bottleneck_size//4, [1, 1],scope=scope+"_grid%d_fc3"%i);
                oy.append( slim.conv2d(x, 3, [1, 1] ,activation_fn = tf.nn.tanh,scope=scope+"_grid%d_fc4"%i) );
            y = tf.concat(oy,axis=2);
            return tf.reshape(y,[tf.shape(siggrid)[0].-1,3]);

def atlas_couple(sig,grid):
    sigexp = tf.reshape(sig,[tf.shape(sig)[0],1,tf.shape(sig)[1]]);
    sig_tiled = tf.tile(sigexp,[1,tf.shape(grid)[1],1]);
    siggrid = tf.concate([grid,sig],axis=2);
    return siggrid;
    
def ATLAS(settings={}):
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
    if 'grid_dim' in settings.keys():
        GRID_DIM=settings['grid_dim'];
    else:
        GRID_DIM=3;
    if 'n_primitives' in settings.keys():
        GRID_NUM = settings['n_primitives'];
    else:
        GRID_NUM = 5;
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
        yGT = tf.placeholder(tf.float32,shape=[None,None,PTS_DIM],name='yGT');
        net['yGT'] = yGT;
        #
        x2D = tf.placeholder(tf.float32,shape=[None,HEIGHT,WIDTH,4],name='x2D');
        net['ix2D'] = x2D;
        #
        xGrid = tf.placeholder(tf.float32,shape=[None,None,GRID_DIM],name='xGrid');
        net['ixGrid'] = xGrid;
        #
        isTrain = tf.placeholder(tf.bool,name='isTrain');
        net['isTrain'] = isTrain;
        #
        shape_sig_ext = tf.placeholder(tf.float32,shape=[None,1024],name='shape_sig_ext');
        net['ixShapeSig'] = shape_sig_ext;
        #
        shape_sig , _ = resnet_v1_18(x2D,num_classes=1024,is_training=isTrain,reuse=reuse);
        print('shape_sig.shape:',shape_sig.shape);
        net['oxShapeSig'] = shape_sig;
        #
        yext = atlas_decode(atlas_couple(shape_sig_ext,xGrid),isTrain,scope="decoder",reuse=reuse);
        net['o3Dext'] = yext;
        y = atlas_decode(atlas_couple(shape_sig,xGrid),isTrain,scope="decoder",reuse=True);
        net['o3D'] = y;
        #    
        dists_forward,_,dists_backward,_ = loss.ChamferDistLoss.Loss(yGT,y);
        dists_forward=tf.reduce_mean(dists_forward);
        dists_backward=tf.reduce_mean(dists_backward);
        tf.summary.scalar("dists_forward",dists_forward);
        tf.summary.scalar("dists_backward",dists_backward);
        #
        loss_nodecay=(dists_forward+dists_backward)*1024*100;
        tf.summary.scalar("loss_no_decay",loss_nodecay);
        net['chmf'] = loss_nodecay;
        #
        decay = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))*0.1;
        tf.summary.scalar("decay",decay);
        #
        loss_with_decay = loss_nodecay + decay;
        tf.summary.scalar("loss_with_decay",loss_with_decay);
        #
        lr  = tf.placeholder(tf.float32,name='lr');
        net['lr'] = lr;
        stepinit = tf.constant_initializer(0);
        gstep = tf.get_variable(shape=[],initializer=stepinit,trainable=False,name='step',dtype=tf.int32);
        net['step'] = gstep;
        optchmf = tf.train.AdamOptimizer(lr).minimize(loss_with_decay,global_step=gstep);
        net['optchmf'] = optchmf;
return net;