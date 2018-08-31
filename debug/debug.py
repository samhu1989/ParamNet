import net;
import sys;
import os;
import shutil;
import tensorflow as tf;
sys.path.append('..');
import util;
import numpy as np;
from data import DataFetcher;
from net import loss;
import time;
def debug_train():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2";
    traindir = "/data4T1/samhu/shapenet_split_complete/train";
    settings = {};
    settings['batch_size'] = 32;
    settings['height'] = 192;
    settings['width'] = 256;
    net_name='KPARAM_33_DUAL'
    net_dict = net.build_model(net_name,settings);
        
    train_fetcher = DataFetcher();
    train_fetcher.BATCH_SIZE = settings['batch_size'];
    train_fetcher.PTS_DIM = 3;
    train_fetcher.HEIGHT = settings['height'];
    train_fetcher.WIDTH = settings['width'];
    train_fetcher.Dir = util.listdir(traindir,".h5");
    train_fetcher.shuffleDir();
    
    if 'rand' in net_dict.keys():
        train_fetcher.randfunc=net_dict['rand'];
    
    config=tf.ConfigProto();
    config.gpu_options.allow_growth = True;
    config.allow_soft_placement = True;
    lrate = 3e-5;
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer());
        try:
            train_fetcher.start();
            lastEpoch = 0;
            data_dict = train_fetcher.fetch();
            x2D = data_dict['x2D'];
            x3D = data_dict['x3D'];
            yGT = 0.8*data_dict['x3D_final']; 
            GT_PTS_NUM = int(yGT.shape[1]);
            print x3D.shape;
            print yGT.shape;
            yGT = yGT.reshape((-1,3));
            x3D = x3D.reshape((-1,3));
            feed={
                 net_dict['yGT']:yGT,
                 net_dict['ix3D']:x3D,
                 net_dict['ix2D']:x2D,
                 net_dict['lr']:lrate
                };
            if 'eidx' in data_dict.keys():
                i = 0;
                while 'eidx_%d'%i in net_dict.keys():
                    feed[net_dict['eidx_%d'%i]] = data_dict['eidx'][i];
                    i += 1;
            if ('fidx' in data_dict.keys()) and ('fidx' in net_dict.keys()):
                feed[net_dict['fidx']] = data_dict['fidx'][-1];
            yout = sess.run(net_dict['ox3D'],feed_dict=feed);
            print yout.shape;
            train_fetcher.shutdown();
        finally:
            train_fetcher.shutdown();
    return;

def debug_up():
    N = 2;
    M = 34;
    level = 3;
    for t in range(3):
        data_dict = util.rand_sphere_interp(N,M,level);
        pts = data_dict['x3D'];
        interpidx = data_dict['eidx'];
        flst = data_dict['fidx'];
        faces = [];
        for j in range(pts.shape[0]):
            faces.append(flst[0]);
        print pts.shape;
        util.write_to_obj('../debug/T%d_L0'%t,pts,faces=faces);
        for i,idx in enumerate(interpidx):
            interp_pts = pts[:,idx,:];
            interp_pts = np.mean(interp_pts,axis=2);
            interp_pts_norm = np.sqrt(np.sum(np.square(interp_pts),axis=2,keepdims=True));
            interp_pts /= interp_pts_norm;
            pts = np.concatenate([pts,interp_pts], axis=1);
            faces = [];
            for j in range(pts.shape[0]):
                faces.append(flst[i+1]);
            print pts.shape;
            util.write_to_obj('../debug/T%d_L%d'%(t,(i+1)),pts,faces=faces);
        faces = [];
        for j in range(pts.shape[0]):
            faces.append(flst[-1]);
        util.write_to_obj('../debug/T%d_final'%t,data_dict['x3D_final'],faces=faces);

def debug_emd():
    xyz1 = np.random.uniform(-1,1,[32,1024,3]).astype(np.float32);
    xyz2 = np.random.uniform(-1,1,[32,1024,3]).astype(np.float32);
    t0 = time.clock();
    print loss.emd.emd_eval(xyz1,xyz2);
    print loss.emd.emd_eval(xyz2,xyz1);
    print time.clock()-t0,"s";
    
def debug_realbatch():
    path = '/data4T/samhu/real';
    util.genRealBatch(path);
    
if __name__ == "__main__":
    debug_realbatch();