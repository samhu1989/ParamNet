import net;
import sys;
import os;
import shutil;
import tensorflow as tf;
import util;
import numpy as np;

FetcherLst = [];

def shutdownall():
    for fetcher in FetcherLst:
        if isinstance(fetcher, DataFetcher):
            fetcher.shutdown();

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
    val_fetcher.WIDTH = settings['height'];
    val_fetcher.Dir = util.listdir(valdir,".h5");
    val_fetcher.shuffleDir();
    
    FetcherLst.append(val_fetcher);
    
    if len(net_model) > 4:
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
            for traincnt in range(10*len(train_fetcher.Dir)):
                out = train_fetcher.fetch();
                x2D = out[0];
                x3D = out[1];
                yGT = out[-1]; 
                GT_PTS_NUM = int(yGT.shape[1]);
                yGT = yGT.reshape((-1,3));
                x3D = x3D.reshape((-1,3));
                feed={
                    net_dict['yGT']:yGT,
                    net_dict['ix3D']:x3D,
                    net_dict['ix2D']:x2D,
                    net_dict['lr']:lrate
                };
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
                x2D,x3D,yGT = val_fetcher.fetch();
                yGT = yGT.reshape((-1,3));
                x3D = x3D.reshape((-1,3));
                feed={
                    net_dict['yGT']:yGT,
                    net_dict['ix3D']:x3D,
                    net_dict['ix2D']:x2D
                };
                summary,loss,step = sess.run([net_dict['sum'],net_dict['chmf'],net_dict['step']],feed_dict=feed);
                valid_writer.add_summary(summary,step);
                if step % 200 == 0:
                    saver.save(sess,'%s/'%dumpdir+"model_epoch%d.ckpt"%train_fetcher.EpochCnt);
                epoch_len = len(train_fetcher.Dir);
                print "Epoch:",train_fetcher.EpochCnt,"GT_PTS_NUM",GT_PTS_NUM,"step:",step,"/",epoch_len,"learning rate:",lrate;
        finally:
            train_fetcher.shutdown();
            val_fetcher.shutdown();
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
            traindir = datadir+"/train";
            testdir = datadir+"/test";
            valdir = datadir+"/val";
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
            test(settings);
        elif cmd=="test":
            test(settings);
        else:
            assert False,"input format wrong";
    finally:
        shutdownall();
        print "ended"