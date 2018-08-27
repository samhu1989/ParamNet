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
            
def train(settings={}):
    if not os.path.exists(dumpdir):
        os.mkdir(dumpdir);
    gpu_num = 1;
    if 'gpu_num' in settings.keys():
        gpu_num = settings['gpu_num'];
    net_dict_lst = [];
    settings['dev']='/gpu:0';
    net_dict_lst.append(net.build_model(net_name,settings));
    tower_grads = [];
    with tf.device( '/cpu:0' ):
        opt = tf.train.AdamOptimizer(net_dict_lst[-1]['lr']);
    for i in range(1,gpu_num):
        settings['dev'] = '/gpu:%d'%i;
        settings['reuse'] = True;
        net_dict_lst.append(net.build_model(net_name,settings));
        grads = opt.compute_gradients(net_dict_lst[-1]['loss_with_decay']);
        tower_grads.append(grads);
    avg_grads = util.average_gradients(tower_grads);
    train_op = opt.apply_gradients(avg_grads,global_step = net_dict_lst[0]['step']);
        
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
            for traincnt in range(0,800,gpu_num):
                feed_all = {}; 
                for i in range(gpu_num):
                    data_dict = train_fetcher.fetch();
                    yGT = data_dict['yGTdense']; 
                    GT_PTS_NUM = int(yGT.shape[1]);
                    feed_all.update(util.generate_feed(data_dict,net_dict_lst[i],lrate));
                _,step,summary = sess.run([train_op,net_dict_lst[0]['step'],net_dict_lst[0]['sum']],feed_dict=feed_all);
                train_writer.add_summary(summary,gpu_num*step);
                epoch_len = len(train_fetcher.Dir);
                n_epoch = (gpu_num*step) // epoch_len;
                if step*gpu_num > 8*len(train_fetcher.Dir):
                    saver.save(sess,'%s/'%dumpdir+"model_epoch%d.ckpt"%n_epoch);
                    break;
                if step % 200 == 0:
                    saver.save(sess,'%s/'%dumpdir+"model_epoch%d.ckpt"%n_epoch);
                if step % 200 == 0:
                    data_dict = valid_fetcher.fetch();
                    feed = util.generate_feed(data_dict,net_dict_lst[0],lrate);
                    summary,loss,step = sess.run([net_dict_lst[0]['sum'],net_dict_lst[0]['chmf'],net_dict_lst[0]['step']],feed_dict=feed);
                    valid_writer.add_summary(summary,step);

                print "Epoch:",n_epoch,"GT_PTS_NUM",GT_PTS_NUM,"step:",step*gpu_num,"/",epoch_len,"learning rate:",lrate;
        finally:
            train_fetcher.shutdown();
            val_fetcher.shutdown();
    return;
            
            
if __name__ == "__main__":
    #some default value
    datadir="/data4T/samhu/shapenet_split_complete";
    dumpdir="/data4T/samhu/tf_dump/SL_Exp_04_train";
    preddir="/data4T/samhu/tf_dump/predict";
    net_name="PSGN";
    gpuid="0";
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
            gpuid = pt[4:];
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
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid;
    settings['gpu_num'] = len(gpuid.split(','));
    try:
        if cmd=="train":
            train(settings);
        else:
            assert False,"input format wrong";
    finally:
        shutdownall();
        print "ended"