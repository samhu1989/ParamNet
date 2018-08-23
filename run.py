import tensorflow as tf;
import argparse;
import sys;
import util;
import os;

if __name__ == '__main__':
    parse = argparse.ArgumentParser();
    parse.add_argument("--data",type=str,default='mnist',help="data path");
    parse.add_argument("--net",type=str,default='PARAM',help="network name");
    parse.add_argument("--dump",type=str,help="dump path where the trained model and other information is saved");
    parse.add_argument("--gpu",type=str,default='0',help="gpu used");
    parse.add_argument("--cmd",type=str,default='train',help="command");
    parse.add_argument("--batch_size",type=int,default=32,help="set batch size");
    parse.add_argument("--test_batch_size",type=int,default=32,help="set test batch size");
    parse.add_argument("--img_width",type=int,default=256,help="set image width");
    parse.add_argument("--img_height",type=int,default=192,help="set image height");
    parse.add_argument("--img_channel",type=int,default=4,help="set image channel");
    flags = parse.parse_args();
    settings={};
    settings['batch_size'] = flags.batch_size;
    settings['test_batch_size'] = flags.test_batch_size;
    settings['img_width'] = flags.img_width;
    settings['img_height'] = flags.img_height;
    settings['img_channel'] = flags.img_channel;
    os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpu;
    try:
        if flags.cmd == 'train':
            util.train(settings);
        elif flags.cmd == 'test':
            util.test(settings);
    except Exception,e:
        print(e);
    else:
        print('done');