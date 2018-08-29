import threading;
import Queue;
import util;
import h5py;
import numpy as np;
import random;
import os;
import sys;
from .sampling import *;
class DataFetcher(threading.Thread):
    def __init__(self):
        super(DataFetcher,self).__init__()
        self.BATCH_SIZE = 16;
        self.HEIGHT=192;
        self.WIDTH=256;
        self.PTS_DIM=3;
        self.Data = Queue.Queue(256);
        self.DataTag = Queue.Queue(256);
        self.Cnt = 0;
        self.EpochCnt = 0;
        self.stopped = False;
        self.Dir = [];
        self.useMix = True;
        self.randfunc="rand_n_sphere(self.BATCH_SIZE,PTS_NUM)";
    
    def shuffleDir(self):
        random.shuffle(self.Dir);
        
    def workMix(self,verb=False):
        q = [];
        files = [];
        cnt = 0;
        PTS_NUM = None;
        PTS_DENSE_NUM = None;
        VIEW_NUM = None;
        while cnt < self.BATCH_SIZE:
            if verb:
                print >>sys.stderr,'reading ',cnt,'/',self.BATCH_SIZE;
            datapath = self.Dir[self.Cnt];
            f = h5py.File(datapath,"r");
            fdense = None;
            densepath = datapath.split(".")[0]+".dense";
            x2DIn = f["IMG"][...];
            yGTIn = f["PV"][...];
            if PTS_NUM is None:
                PTS_NUM = int(yGTIn.shape[-2]);
            else:
                assert PTS_NUM==int(yGTIn.shape[-2]);
            if os.path.exists(densepath):
                fdense = h5py.File(densepath,"r");
                yGTDenseIn = fdense["PV"][...];
                if PTS_DENSE_NUM is None :
                    PTS_DENSE_NUM= int(yGTDenseIn.shape[-2]);
                else:
                    assert PTS_DENSE_NUM==int(yGTDenseIn.shape[-2]);
            if VIEW_NUM is None:
                VIEW_NUM = int(yGTIn.shape[0]);
            if VIEW_NUM is None:
                VIEW_NUM = int(yGTIn.shape[0]);
            else:
                assert VIEW_NUM == int(yGTIn.shape[0]);
            if not np.isfinite(x2DIn).all():
                print >>sys.stderr,datapath," contain invalid data in x2D";
            elif not np.isfinite(yGTIn).all():
                print >>sys.stderr,datapath," contain invalid data in yGT";
            else:
                files.append((f,fdense));
                cnt += 1;
            self.Cnt += 1;
            if self.Cnt >= len(self.Dir):
                self.Cnt = 0;
                self.EpochCnt += 1;
        for i in range(VIEW_NUM):
            if verb:
                print >>sys.stderr,'allocating ',i,'/',VIEW_NUM;
            x2D = np.zeros([self.BATCH_SIZE,self.HEIGHT,self.WIDTH,4]);
            rand = eval(self.randfunc);
            yGT = np.zeros([self.BATCH_SIZE,PTS_NUM,3]);
            data_dict = {};
            data_dict['x2D'] = x2D;
            data_dict['yGT'] = yGT;
            data_dict.update(rand);
            if fdense is not None:
                yGTdense = np.zeros([self.BATCH_SIZE,PTS_DENSE_NUM,3]);
                data_dict['yGTdense'] = yGTdense;
                data_dict['ynGT'] = yGTdense.copy();
            q.append(data_dict);
        fi = 0;
        for f,fdense in files:
            if verb:
                print >>sys.stderr,'reading dense', fi,'/',len(files) ;
            x2DIn = f["IMG"][...];
            yGTIn = f["PV"][...];
            yGTDense = None;
            if fdense is not None:
                yGTDense = fdense["PV"][...];
                ynGTDense = fdense["PN"][...];
            for i in range(VIEW_NUM):
                q[i]['x2D'][fi,...] = x2DIn[i,...];
                q[i]['yGT'][fi,...] = yGTIn[i,...];
                if yGTDense is not None:
                    q[i]['yGTdense'][fi,...] = yGTDense[i//2,...];
                    q[i]['ynGT'][fi,...] = ynGTDense[i//2,...];
            f.close();
            if fdense:
                fdense.close();
            fi += 1;
        return q;
    
    def workNoMix(self,verb=False):
        q = [];
        tag = [];
        datapath = self.Dir[self.Cnt];
        f = h5py.File(datapath,"r");
        ftag = os.path.basename(datapath).split("_")[0];
        densepath = datapath.split(".")[0]+".dense";
        fdense = None;
        if os.path.exists(densepath):
            fdense = h5py.File(densepath,"r");
            yGTDenseIn = fdense["PV"][...];
            ynGTDenseIn = fdense["PN"][...];
            PTS_DENSE_NUM = int(yGTDenseIn.shape[-2]);
        self.Cnt += 1;
        x2DIn = f["IMG"];
        yGTIn = f["PV"];
        VIEW_NUM = int(yGTIn.shape[0]);
        PTS_NUM = int(yGTIn.shape[-2]);
        if self.Cnt >= len(self.Dir):
            self.Cnt = 0;
            self.EpochCnt += 1;
        num = VIEW_NUM // self.BATCH_SIZE;
        assert num*self.BATCH_SIZE==VIEW_NUM,"self.BATCH_SIZE is not times of VIEW_NUM in dataset";
        data_dict = {};
        for i in range(num):
            x2D = np.zeros([self.BATCH_SIZE,self.HEIGHT,self.WIDTH,4]);
            rand = eval(self.randfunc);
            yGT = np.zeros([self.BATCH_SIZE,PTS_NUM,3]);
            data_dict={};
            data_dict['x2D'] = x2D;
            data_dict['yGT'] = yGT;
            data_dict.update(rand);
            if fdense:
                data_dict['yGTdense'] = np.zeros([self.BATCH_SIZE,PTS_DENSE_NUM,3]);
                data_dict['ynGT'] = np.zeros([self.BATCH_SIZE,PTS_DENSE_NUM,3]);
            q.append(data_dict);
            tag.append(ftag);
        for i in range(VIEW_NUM):
            qi = i // self.BATCH_SIZE ;
            qj = i % self.BATCH_SIZE;
            q[qi]['x2D'][qj,...] = x2DIn[i,...];
            q[qi]['yGT'][qj,...] = yGTIn[i,...];
            if fdense:
                q[qi]['yGTdense'][qj,...] = yGTDenseIn[i//2,...];
                q[qi]['ynGT'][qj,...] = ynGTDenseIn[i//2,...];
        f.close();
        if fdense:
            fdense.close();
        return q,tag;
    
    def run(self):
        while not self.stopped:
            if self.Data.empty():
                verb = False;
            else:
                verb = False;
            if self.Dir is not None:
                q = [];
                tags = [];
                if self.useMix:
                    q = self.workMix(verb);
                else:
                    q,tags = self.workNoMix(verb);
                if verb:
                    print >>sys.stderr,'putting into data';
                for v in q:
                    self.Data.put(v);
                for tag in tags:
                    self.DataTag.put(tag);
                if verb:
                    print >>sys.stderr,'Done put into data';
    
    def fetch(self):
        if self.stopped:
            return None;
        return self.Data.get();
    
    def fetchTag(self):
        if self.stopped:
            return None;
        return self.DataTag.get();
    
    def shutdown(self):
        self.stopped=True;
        while not self.Data.empty():
            self.Data.get();
        while not self.DataTag.empty():
            self.DataTag.get();
            