import net;
import sys;
import os;
import shutil;
import tensorflow as tf;
import util;
import numpy as np;
from data import DataFetcher;
try:
  import cPickle as pickle;
except ImportError:
  import pickle ;
from tensorflow.python import pywrap_tensorflow;
import scipy;

FetcherLst = [];

def shutdownall():
    for fetcher in FetcherLst:
        if isinstance(fetcher, DataFetcher):
            fetcher.shutdown();

def testlayers(settings={}):
    if not os.path.exists(preddir):
        os.mkdir(preddir);
    net_model = None;
    config = None;
    if not os.environ["CUDA_VISIBLE_DEVICES"]:
        net_dict = net.build_model(net_name,settings);
        config = tf.ConfigProto(intra_op_parallelism_threads=4,device_count={'gpu':0});
    else:
        net_dict = net.build_model(net_name,settings);
        config = tf.ConfigProto();
        config.gpu_options.allow_growth = True;
        config.allow_soft_placement = True;
        
    test_fetcher = DataFetcher();
    test_fetcher.BATCH_SIZE = settings['batch_size'];
    test_fetcher.PTS_DIM = 3;
    test_fetcher.HEIGHT = settings['height'];
    test_fetcher.WIDTH = settings['width'];
    test_fetcher.Dir = util.listdir(traindir,".h5");
    test_fetcher.useMix = False;
    
    FetcherLst.append(test_fetcher);
    
    if 'rand' in net_dict.keys():
        test_fetcher.randfunc=net_dict['rand'];
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer());
        saver = tf.train.Saver();
        ckpt = tf.train.get_checkpoint_state('%s/'%dumpdir);
        if ckpt and ckpt.model_checkpoint_path:  
            saver.restore(sess, ckpt.model_checkpoint_path);
        else:
            print "failed to restore model";
            return;
        layers = [];
        li = 0;
        while 'ox3D%02d'%li in net_dict.keys():
            layers.append(net_dict['ox3D%02d'%li]);
            li += 1;
        cnt = 30;
        test_fetcher.Cnt = cnt;
        try:
            test_fetcher.start();
            data_dict = test_fetcher.fetch();
            x2D = data_dict['x2D'];
            x3D = data_dict['x3D'];
            yGT = data_dict['yGT'];
            tag = test_fetcher.fetchTag();
                #
            yGTout = yGT.copy();
            yGT = yGT.reshape((-1,3));
            x3D = x3D.reshape((-1,3));
            feed={
                net_dict['yGT']:yGT,
                net_dict['ix3D']:x3D,
                net_dict['ix2D']:x2D
            };
            if 'eidx' in data_dict.keys():
                i = 0;
            while 'eidx_%d'%i in net_dict.keys():
                feed[net_dict['eidx_%d'%i]] = data_dict['eidx'][i];
                i += 1;
            if ('fidx' in data_dict.keys()) and ('fidx' in net_dict.keys()):
                feed[net_dict['fidx']] = data_dict['fidx'][-1];
            ylayers = sess.run(layers,feed_dict=feed);
            fdir = preddir+os.sep+"pred_%s_%03d"%(tag,cnt);
            if not os.path.exists(fdir):
                os.mkdir(fdir);
            i = 0;
            for layer in layers:
                lname = layer.name.replace(':','_');
                if i < len(data_dict['fidx']):
                    fidx = data_dict['fidx'][i];
                else:
                    fidx = data_dict['fidx'][-1];
                f_lst = [];
                for j in range(test_fetcher.BATCH_SIZE):
                    f_lst.append(fidx);
                util.write_to_obj(fdir+os.sep+"obj%02d"%i,ylayers[i],faces=f_lst);
                i += 1;
        finally:
            test_fetcher.shutdown();
    return;

def train(settings={}):
    if not os.path.exists(preddir):
        os.mkdir(preddir);
    net_model = None;
    config = None;
    if not os.environ["CUDA_VISIBLE_DEVICES"]:
        net_dict = net.build_model(net_name,settings);
        config = tf.ConfigProto(intra_op_parallelism_threads=4,device_count={'gpu':0});
    else:
        net_dict = net.build_model(net_name,settings);
        config = tf.ConfigProto();
        config.gpu_options.allow_growth = True;
        config.allow_soft_placement = True;
        
    train_fetcher = DataFetcher();
    train_fetcher.BATCH_SIZE = settings['batch_size'];
    train_fetcher.PTS_DIM = 3;
    train_fetcher.HEIGHT = settings['height'];
    train_fetcher.WIDTH = settings['width'];
    train_fetcher.Dir = util.listdir(traindir,".h5");
    train_fetcher.useMix = False;
    
    FetcherLst.append(train_fetcher);
    
    if 'rand' in net_dict.keys():
        train_fetcher.randfunc=net_dict['rand'];
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer());
        saver = tf.train.Saver();
        ckpt = tf.train.get_checkpoint_state('%s/'%dumpdir);
        if ckpt and ckpt.model_checkpoint_path:  
            assign = util.assign_from_checkpoint_fn(ckpt.model_checkpoint_path,tf.all_variables(),True);
            assign(sess);
        else:
            print "failed to restore model";
            return;
        stat = {};
        try:
            train_fetcher.Cnt = 30;
            train_fetcher.start();
            data_dict = train_fetcher.fetch();
            tag = train_fetcher.fetchTag();
            lrate = 1e-5;
            if 'yGTdense' in net_dict.keys():
                yGT = data_dict['yGT'];
            else:
                yGT = data_dict['yGTdense'];
            x2Dfix = data_dict['x2D'].copy();
            x2Dfix[:,:,:] = data_dict['x2D'][14,:,:];
            x3Dfix = data_dict['x3D'].copy();
            x3Dfix[:,:,:] = data_dict['x3D'][14,:,:];
            yGTfix = yGT.copy();
            yGTfix[:,:,:] = yGT[14,:,:];
            
            for cnt in range(1000):
                x2D = x2Dfix;
                x3D = x3Dfix;
                yGT = yGTfix;
                r2D = None;
                if 'r2D' in net_dict.keys():
                    r2D_dim = int(net_dict['r2D'].shape[1]);
                    r2D = np.random.normal(loc=0.0,scale=1.0,size=[BATCH_SIZE,r2D_dim]);
                #
                rgb = None;
                f_lst = [];
                if 'x3D_final' in data_dict.keys():
                    rgb = util.sphere_to_YIQ( data_dict['x3D_final'] );
                #\
                print cnt,int(yGT.shape[1]);
                yGTout = yGT.copy();
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
                yout=None;
                loss=None;
                _,yout,loss = sess.run([net_dict['optchmf'],net_dict['ox3D'],net_dict['chmf']],feed_dict=feed);
                fdir = preddir+os.sep+"pred_%s_%03d_%f"%(tag,cnt,loss);
                if cnt%100==0:
                #
                    if not os.path.exists(fdir):
                        os.mkdir(fdir);
                    if 'fidx' in data_dict.keys():
                        for j in range(yout.shape[0]):
                            f_lst.append(data_dict['fidx'][-1]);
                        util.write_to_obj(fdir+os.sep+"obj",yout,rgb,f_lst);
                    else:
                        util.write_to_obj(fdir+os.sep+"obj",yout);
                    util.write_to_obj(fdir+os.sep+"GTobj",yGTout);
                    #generating dense result
                    util.write_to_img(fdir,x2D);
                    saver.save(sess,'%s/'%dumpdir+"model_epoch%d.ckpt"%cnt);
        finally:
            train_fetcher.shutdown();
    return;
    

def test(settings={}):
    if not os.path.exists(preddir):
        os.mkdir(preddir);
    net_model = None;
    config = None;
    if not os.environ["CUDA_VISIBLE_DEVICES"]:
        net_dict = net.build_model(net_name,settings);
        config = tf.ConfigProto(intra_op_parallelism_threads=4,device_count={'gpu':0});
    else:
        net_dict = net.build_model(net_name,settings);
        config = tf.ConfigProto();
        config.gpu_options.allow_growth = True;
        config.allow_soft_placement = True;
        
    test_fetcher = DataFetcher();
    test_fetcher.BATCH_SIZE = settings['batch_size'];
    test_fetcher.PTS_DIM = 3;
    test_fetcher.HEIGHT = settings['height'];
    test_fetcher.WIDTH = settings['width'];
    test_fetcher.Dir = util.listdir(traindir,".h5");
    test_fetcher.useMix = False;
    
    FetcherLst.append(test_fetcher);
    
    if 'rand' in net_dict.keys():
        test_fetcher.randfunc=net_dict['rand'];
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer());
        saver = tf.train.Saver();
        ckpt = tf.train.get_checkpoint_state('%s/'%dumpdir);
        if ckpt and ckpt.model_checkpoint_path:  
            saver.restore(sess, ckpt.model_checkpoint_path);
        else:
            print "failed to restore model";
            return;
        stat = {};
        try:
            test_fetcher.start();
            for cnt in range(len(test_fetcher.Dir)):
                data_dict = test_fetcher.fetch();
                x2D = data_dict['x2D'];
                x3D = data_dict['x3D'];
                yGT = data_dict['yGT'];
                tag = test_fetcher.fetchTag();
                r2D = None;
                if 'r2D' in net_dict.keys():
                    r2D_dim = int(net_dict['r2D'].shape[1]);
                    r2D = np.random.normal(loc=0.0,scale=1.0,size=[BATCH_SIZE,r2D_dim]);
                #
                rgb = None;
                f_lst = [];
                if 'x3D_final' in data_dict.keys():
                    rgb = util.sphere_to_YIQ( data_dict['x3D_final'] );
                #
                yGTout = yGT.copy();
                yGT = yGT.reshape((-1,3));
                x3D = x3D.reshape((-1,3));
                feed={
                    net_dict['yGT']:yGT,
                    net_dict['ix3D']:x3D,
                    net_dict['ix2D']:x2D
                };
                if 'eidx' in data_dict.keys():
                    i = 0;
                    while 'eidx_%d'%i in net_dict.keys():
                        feed[net_dict['eidx_%d'%i]] = data_dict['eidx'][i];
                        i += 1;
                if ('fidx' in data_dict.keys()) and ('fidx' in net_dict.keys()):
                    feed[net_dict['fidx']] = data_dict['fidx'][-1];
                yout=None;
                loss=None;
                yout,loss = sess.run([net_dict['ox3D'],net_dict['chmf']],feed_dict=feed);
                fdir = preddir+os.sep+"pred_%s_%03d_%f"%(tag,cnt,loss);
                #
                if not os.path.exists(fdir):
                    os.mkdir(fdir);
                if 'fidx' in data_dict.keys():
                    for j in range(yout.shape[0]):
                        f_lst.append(data_dict['fidx'][-1]);
                    util.write_to_obj(fdir+os.sep+"obj",yout,rgb,f_lst);
                else:
                    util.write_to_obj(fdir+os.sep+"obj",yout);
                util.write_to_obj(fdir+os.sep+"GTobj",yGTout);
                #generating dense result
                util.write_to_img(fdir,x2D);
        finally:
            test_fetcher.shutdown();
    return;

if __name__ == "__main__":
    #some default value
    datadir="/data4T1/samhu/shapenet_split_complete";
    dumpdir="/data4T1/samhu/tf_dump/SL_Exp_04_train";
    preddir="/data4T1/samhu/tf_dump/predict";
    net_name="VPSGN";
    gpuid=1;
    testbegin = None;
    testend = None;
    for pt in sys.argv[1:]:
        if pt[:5]=="data=":
            datadir = pt[5:];
        elif pt[:5]=="dump=":
            dumpdir = pt[5:];
        elif pt[:5]=="pred=":
            preddir = pt[5:];
        elif pt[:4]=="gpu=":
            gpuid = int(pt[4:]);
        elif pt[:4]=="net=":
            net_name = pt[4:];
        else:
            cmd = pt;
    preddir += "/" + net_name;
    dumpdir += "/" + net_name;
    traindir = datadir+"/train";
    testdir = datadir+"/test";
    valdir = datadir+"/val";
    settings = {};
    settings['batch_size'] = 32;
    settings['height'] = 192;
    settings['width'] = 256;
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%gpuid;
    try:
        if cmd=="testlayers":
            testlayers(settings);
        elif cmd=="train":
            os.environ["CUDA_VISIBLE_DEVICES"]="1";
            settings['dev']='/gpu:0';
            train(settings);
        elif cmd=="cputest":
            os.environ["CUDA_VISIBLE_DEVICES"]="";
            settings['dev']='/cpu:0';
            test(settings);
        elif cmd=="test":
            test(settings);
        else:
            assert False,"input format wrong";
    finally:
        shutdownall();
        print "ended"