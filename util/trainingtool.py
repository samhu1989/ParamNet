import tensorflow as tf;
from data import DataFetcher;
import os;
import sys;
sys.path.append('..');
import net;
from .io import listdir;
from .safety import safe_guard;
import datetime;
from tensorflow.python import pywrap_tensorflow;

def assign_from_checkpoint_fn(model_path, var_list, ignore_missing_vars=False,
                              reshape_variables=False):
    if ignore_missing_vars:
        reader = pywrap_tensorflow.NewCheckpointReader(model_path)
    if isinstance(var_list, dict):
        var_dict = var_list
    else:
        var_dict = {var.op.name: var for var in var_list}
    available_vars = {}
    for var in var_dict:
        if reader.has_tensor(var):
            available_vars[var] = var_dict[var]
        else:
            tf.logging.warning('Variable %s missing in checkpoint %s', var, model_path)
    var_list = available_vars
    saver = tf.train.Saver(var_list, reshape=reshape_variables)
    def callback(session):
        saver.restore(session, model_path);
    return callback

def average_gradients(tower_grads):
    """Calculate average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been
       averaged across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0);
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1];
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads;

def generate_feed(data_dict,net_dict,istrain,lrate):
    x2D = data_dict['x2D'];
    x3D = data_dict['xgrid'];
    if 'yGTdense' in data_dict.keys():
        yGT = data_dict['yGTdense'];
    else:
        yGT = data_dict['yGT'];
    feed={
        net_dict['yGT']:yGT,
        net_dict['ixGrid']:x3D,
        net_dict['ix2D']:x2D,
        net_dict['isTrain']:istrain,
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
    return feed;
            
def build(settings={}):
    gpu_num = len( os.environ['CUDA_VISIBLE_DEVICES'].split(',') );
    settings['dev']='/gpu:0';
    net_dict_lst = [];
    net_dict_lst.append(net.build_model(settings['net'],settings));
    train_op = net_dict_lst[-1]['opt'+settings['loss']];
    if gpu_num > 1:
        tower_grads = [];
        with tf.device('/cpu:0'):
            opt = tf.train.AdamOptimizer(net_dict_lst[0]['lr']);
        with tf.device(settings['dev']):
            grads = opt.compute_gradients(net_dict_lst[0][settings['loss']]);
        tower_grads.append(grads);
        for i in range(1,gpu_num):
            settings['dev'] = '/gpu:%d'%i;
            settings['reuse'] = True;
            net_dict_lst.append(net.build_model(settings['net'],settings));
            with tf.device(settings['dev']):
                grads = opt.compute_gradients(net_dict_lst[0][settings['loss']]);
            tower_grads.append(grads);
        with tf.device( '/gpu:0' ):
            avg_grads = average_gradients(tower_grads);
        with tf.device( '/cpu:0' ):
            train_op = opt.apply_gradients(avg_grads,global_step = net_dict_lst[0][settings['step']]);
    print(settings['net']+' built on %d gpus'%gpu_num);
    return net_dict_lst,train_op;

def generate_fetcher(settings,subpath,shuffle=True):
    fetcher = DataFetcher();
    fetcher.BATCH_SIZE = settings['batch_size'];
    fetcher.PTS_DIM = 3;
    fetcher.HEIGHT = settings['height'];
    fetcher.WIDTH = settings['width'];
    fetcher.Dir = listdir(settings['data']+os.sep+subpath,".h5");
    if shuffle:
        fetcher.shuffleDir();
    return fetcher;
    
def train(settings={}):
    if not os.path.exists(settings['dump']):
        os.mkdir(settings['dump']);
    net_dicts,train_op = build(settings);
    gpu_num = len( net_dicts );
    
    train_fetcher = generate_fetcher(settings,'train');
    valid_fetcher = generate_fetcher(settings,'val');
    
    if 'rand' in net_dicts[0].keys():
        train_fetcher.randfunc=net_dicts[0]['rand'];
        valid_fetcher.randfunc=net_dicts[0]['rand'];
        
    try:
        #start fetching data into memory
        train_fetcher.start();
        valid_fetcher.start();
        #define session
        config=tf.ConfigProto();
        config.gpu_options.allow_growth = True;
        config.allow_soft_placement = True;
        
        lrate = 3e-5;    
        with tf.Session(config=config) as sess:
            #restore and save model 
            sess.run(tf.global_variables_initializer());
            ckpt = tf.train.get_checkpoint_state(settings['dump']+os.sep);
            if ckpt and ckpt.model_checkpoint_path:
                assign = assign_from_checkpoint_fn(ckpt.model_checkpoint_path,tf.global_variables(),True);
                assign(sess);
            saver = tf.train.Saver();
            #summary writer
            train_writer = tf.summary.FileWriter(settings['dump']+os.sep+'train',graph=sess.graph);
            valid_writer = tf.summary.FileWriter(settings['dump']+os.sep+'valid',graph=sess.graph);
            #start iteration
            step = sess.run(net_dicts[0][settings['step']]);
            print >> sys.stderr,datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S");
            while step*gpu_num < settings['epoch_num']*len(train_fetcher.Dir):
                cnt = gpu_num*step;
                epoch_len = len(train_fetcher.Dir);
                n_epoch = cnt // epoch_len;
                print >> sys.stderr,"Start Step:",cnt,"/",epoch_len;
                feed_all = {};
                for i in range(gpu_num):
                    data_dict = train_fetcher.fetch();
                    yGT = data_dict['yGTdense']; 
                    GT_PTS_NUM = int(yGT.shape[1]);
                    feed_all.update(generate_feed(data_dict,net_dicts[i],True,lrate));
                _,step,summary = sess.run(
                    [
                        train_op,
                        net_dicts[0][settings['step']],
                        net_dicts[0]['sum']
                    ],feed_dict=feed_all);
                train_writer.add_summary(summary,cnt);
                if step % (6//gpu_num) == 0:
                    data_dict = valid_fetcher.fetch();
                    feed = generate_feed(data_dict,net_dicts[0],False,lrate);
                    summary,loss,step = sess.run(
                        [
                            net_dicts[0]['sum'],
                            net_dicts[0][settings['loss']],
                            net_dicts[0][settings['step']]
                        ],feed_dict=feed);
                    valid_writer.add_summary(summary,cnt);
                    print >> sys.stderr,"Done Valid:",cnt,"/",epoch_len;
                if cnt % 400 == 0:
                    safe_guard();
                    saver.save(sess,settings['dump']+os.sep+settings['net']+'_epoch_%d'%n_epoch);
                print >> sys.stderr,"Done Step:",cnt,"/",epoch_len,"Epoch:",n_epoch,"GT_PTS_NUM",GT_PTS_NUM,"learning rate:",lrate;
            saver.save(sess,settings['dump']+os.sep+settings['net']+'_epoch_%d'%n_epoch);
            print >> sys.stderr,datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S");
    except Exception,e:
        print("Exception Raised");
        print(e);
        train_fetcher.shutdown();
        valid_fetcher.shutdown();
    else:
        print("Done Training");
    return;

def test(settings={}):
    return;
    
    