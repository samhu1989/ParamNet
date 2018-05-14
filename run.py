import net;
import sys;
import os;
import shutil;
import tensorflow as tf;
import util;
import numpy as np;
from data import DataFetcher;

FetcherLst = [];

def shutdownall():
    for fetcher in FetcherLst:
        if isinstance(fetcher, DataFetcher):
            fetcher.shutdown();
            
def pretrain(setttings={}):
    if not os.path.exists(dumpdir):
        os.mkdir(dumpdir);
    net_dict = net.build_model(net_name,settings);
        
    train_fetcher = DataFetcher();
    train_fetcher.BATCH_SIZE = settings['batch_size'];
    train_fetcher.PTS_DIM = 3;
    train_fetcher.HEIGHT = settings['height'];
    train_fetcher.WIDTH = settings['width'];
    train_fetcher.Dir = util.listdir(traindir,".h5");
    train_fetcher.shuffleDir();
    
    FetcherLst.append(train_fetcher);
    
    if 'rand' in net_dict.keys():
        train_fetcher.randfunc=net_dict['rand'];
    
    config=tf.ConfigProto();
    config.gpu_options.allow_growth = True;
    config.allow_soft_placement = True;
    saver = tf.train.Saver();
    lrate = 3e-5;
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer());
        train_writer = tf.summary.FileWriter("%s/pretrain"%(dumpdir),graph=sess.graph);
        ckpt = tf.train.get_checkpoint_state('%s/'%dumpdir);
        if ckpt and ckpt.model_checkpoint_path:
            assign = util.assign_from_checkpoint_fn(ckpt.model_checkpoint_path,tf.all_variables(),True);
            assign(sess);
        try:
            train_fetcher.start();
            lastEpoch = 0;
            for traincnt in range(len(train_fetcher.Dir)):
                data_dict = train_fetcher.fetch();
                x2D = data_dict['x2D'];
                x3D = data_dict['x3D'];
                yGT = 0.5*data_dict['x3D_final']+np.array([0,0,0.5],dtype=np.float32); 
                GT_PTS_NUM = int(yGT.shape[1]);
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
                if 'lplidx' in data_dict.keys():
                    i = 0;
                    while 'lplidx_%d'%i in net_dict.keys():
                        feed[net_dict['lplidx_%d'%i]] = data_dict['lplidx'][i];
                        feed[net_dict['lplw_%d'%i]] = data_dict['lplw'][i];
                        i += 1;
                #feed[net_dict['']]
                opt = net_dict['optmse'];
                _,summary,step = sess.run([
                    opt,
                    net_dict['presum'],
                    net_dict['prestep']
                ],feed_dict=feed);
                train_writer.add_summary(summary,step);
                if step % 200 == 0:
                    saver.save(sess,'%s/'%dumpdir+"model_pretrain");
                epoch_len = len(train_fetcher.Dir);
                print "Pretrain,GT_PTS_NUM",GT_PTS_NUM,"step:",step,"/",epoch_len,"learning rate:",lrate;
                if step > len(train_fetcher.Dir):
                    break;
        finally:
            train_fetcher.shutdown();
    return;

def train(setttings={}):
    if not os.path.exists(dumpdir):
        os.mkdir(dumpdir);
    net_dict = net.build_model(net_name,settings);
        
    train_fetcher = DataFetcher();
    train_fetcher.BATCH_SIZE = settings['batch_size'];
    train_fetcher.PTS_DIM = 3;
    train_fetcher.HEIGHT = settings['height'];
    train_fetcher.WIDTH = settings['width'];
    train_fetcher.Dir = util.listdir(traindir,".h5");
    train_fetcher.shuffleDir();
    
    FetcherLst.append(train_fetcher);
    
    val_fetcher = DataFetcher();
    val_fetcher.BATCH_SIZE = settings['batch_size'];
    val_fetcher.PTS_DIM = 3;
    val_fetcher.HEIGHT = settings['height'];
    val_fetcher.WIDTH = settings['width'];
    val_fetcher.Dir = util.listdir(valdir,".h5");
    val_fetcher.shuffleDir();
    
    FetcherLst.append(val_fetcher);
    
    if 'rand' in net_dict.keys():
        train_fetcher.randfunc=net_dict['rand'];
        val_fetcher.randfunc=net_dict['rand'];
    
    config=tf.ConfigProto();
    config.gpu_options.allow_growth = True;
    config.allow_soft_placement = True;
    saver = tf.train.Saver();
    lrate = 3e-5;
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer());
        train_writer = tf.summary.FileWriter("%s/train"%(dumpdir),graph=sess.graph);
        valid_writer = tf.summary.FileWriter("%s/valid"%(dumpdir),graph=sess.graph)
        ckpt = tf.train.get_checkpoint_state('%s/'%dumpdir);
        if ckpt and ckpt.model_checkpoint_path:
            assign = util.assign_from_checkpoint_fn(ckpt.model_checkpoint_path,tf.all_variables(),True);
            assign(sess);
        try:
            train_fetcher.start();
            val_fetcher.start();
            lastEpoch = 0;
            for traincnt in range(8*len(train_fetcher.Dir)):
                data_dict = train_fetcher.fetch();
                x2D = data_dict['x2D'];
                x3D = data_dict['x3D'];
                yGT = data_dict['yGTdense']; 
                GT_PTS_NUM = int(yGT.shape[1]);
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
                if 'lplidx' in data_dict.keys():
                    i = 0;
                    while 'lplidx_%d'%i in net_dict.keys():
                        feed[net_dict['lplidx_%d'%i]] = data_dict['lplidx'][i];
                        feed[net_dict['lplw_%d'%i]] = data_dict['lplw'][i];
                        i += 1;
                #feed[net_dict['']]
                if net_name.endswith('_DUAL'):
                    opt = net_dict['optdchmf'];
                else:
                    opt = net_dict['optchmf'];
                _,summary,step = sess.run([
                    opt,
                    net_dict['sum'],
                    net_dict['step']
                ],feed_dict=feed);
                train_writer.add_summary(summary,step);
                data_dict = val_fetcher.fetch();
                x2D = data_dict['x2D'];
                yGT = data_dict['yGT'];
                x3D = data_dict['x3D'];
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
                if 'lplidx' in data_dict.keys():
                    i = 0;
                    while 'lplidx_%d'%i in net_dict.keys():
                        feed[net_dict['lplidx_%d'%i]] = data_dict['lplidx'][i];
                        feed[net_dict['lplw_%d'%i]] = data_dict['lplw'][i];
                        i += 1;
                summary,loss,step = sess.run([net_dict['sum'],net_dict['chmf'],net_dict['step']],feed_dict=feed);
                valid_writer.add_summary(summary,step);
                if step % 200 == 0:
                    saver.save(sess,'%s/'%dumpdir+"model_epoch%d.ckpt"%train_fetcher.EpochCnt);
                epoch_len = len(train_fetcher.Dir);
                print "Epoch:",train_fetcher.EpochCnt,"GT_PTS_NUM",GT_PTS_NUM,"step:",step,"/",epoch_len,"learning rate:",lrate;
                if step > 8*len(train_fetcher.Dir):
                    break;
        finally:
            train_fetcher.shutdown();
            val_fetcher.shutdown();
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
                if 'lplidx' in data_dict.keys():
                    i = 0;
                    while 'lplidx_%d'%i in net_dict.keys():
                        feed[net_dict['lplidx_%d'%i]] = data_dict['lplidx'][i];
                        feed[net_dict['lplw_%d'%i]] = data_dict['lplw'][i];
                        i += 1;
                yout=None;
                loss=None;
                yout,loss = sess.run([net_dict['ox3D'],net_dict['chmf']],feed_dict=feed);
                fdir = preddir+os.sep+"pred_%s_%03d"%(tag,cnt);
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
                if tag in stat:
                    newcnt = stat[tag+"_cnt"] + 1;
                    stat[tag] = stat[tag]*stat[tag+"_cnt"]/newcnt + loss/newcnt;
                    stat[tag+"_cnt"] = newcnt;
                else:
                    stat[tag] = loss;
                    stat[tag+"_cnt"] = 1.0;
                print "testing:tag=",tag,"loss=",loss,"mean loss of tag=",stat[tag];
                #generating dense result
                util.write_to_img(fdir,x2D);
            f = open(preddir+os.sep+"log.txt","w");
            for (k,v) in stat.items():
                print >>f,k,v;
            f.close();
        finally:
            test_fetcher.shutdown();
    return;


if __name__ == "__main__":
    #some default value
    datadir="/data4T/samhu/shapenet_split_complete";
    dumpdir="/data4T/samhu/tf_dump/SL_Exp_04_train";
    preddir="/data4T/samhu/tf_dump/predict";
    net_name="PSGN";
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
        if cmd=="train":
            train(settings);
        elif cmd=="pretrain":
            pretrain(settings);
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