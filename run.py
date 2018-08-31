import tensorflow as tf;
import argparse;
import sys;
import util;
import os;

if __name__ == '__main__':
    parse = argparse.ArgumentParser();
    parse.add_argument("-d","--data",type=str,default='mnist',help="data path");
    parse.add_argument("-nn","--net",type=str,default='PARAM',help="network name");
    parse.add_argument("-l","--loss",type=str,default='chmf',help="loss name");
    parse.add_argument("-tp","--step",type=str,default='step',help="step name to support separate step num");
    parse.add_argument("-dp","--dump",type=str,help="dump path where the trained model and other info is saved");
    parse.add_argument("-b","--batch_size",type=int,default=32,help="set batch size");
    parse.add_argument("-tb","--test_batch_size",type=int,default=32,help="set test batch size");
    parse.add_argument("-iw","--img_width",type=int,default=256,help="set image width");
    parse.add_argument("-ih","--img_height",type=int,default=192,help="set image height");
    parse.add_argument("-ic","--img_channel",type=int,default=4,help="set image channel");
    parse.add_argument("-pn","--points_num",type=int,default=2500,help="set output points num");
    parse.add_argument("-gn","--grid_num",type=int,default=1,help="set number of grid");
    parse.add_argument("-gd","--grid_dim",type=int,default=3,help="set number of grid");
    parse.add_argument("-in","--epoch_num",type=int,default=10,help="epoch");
    parse.add_argument("-g","--gpu",type=str,help="gpu used");
    parse.add_argument("-c","--cmd",type=str,default='train',help="command");
    flags = parse.parse_args();
    settings={};
    settings['data'] = flags.data;
    settings['net'] = flags.net;
    settings['loss'] = flags.loss;
    settings['step'] = flags.step;
    settings['dump'] = flags.dump+os.sep+flags.net;
    settings['batch_size'] = flags.batch_size;
    settings['test_batch_size'] = flags.test_batch_size;
    settings['width'] = flags.img_width;
    settings['height'] = flags.img_height;
    settings['channel'] = flags.img_channel;
    settings['pts_num'] = flags.points_num;
    settings['grid_num'] = flags.grid_num;
    settings['grid_dim'] = flags.grid_dim;
    settings['epoch_num'] = flags.epoch_num;
    if flags.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpu;
    try:
        if flags.cmd == 'train':
            util.train(settings);
        elif flags.cmd == 'test':
            util.test(settings);
    except Exception,e:
        print(e);
    else:
        print('Done Runing');