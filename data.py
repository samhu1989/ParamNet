import threading;
import Queue;
import util;
import h5py;
import numpy as np;
import random;
import os;
class DataFetcher(threading.Thread):
    def __init__(self):
        super(DataFetcher,self).__init__()
        self.BATCH_SIZE = 16;
        self.HEIGHT=192;
        self.WIDTH=256;
        self.HID_NUM=128;
        self.PTS_DIM=3;
        self.Data = Queue.Queue(64);
        self.DataTag = Queue.Queue(64);
        self.Cnt = 0;
        self.EpochCnt = 0;
        self.stopped = False;
        self.Dir = [];
        self.useMix = True;
        self.randfunc="np.random.normal(0.0,1.0,[self.HID_NUM,PTS_NUM,3])";
    
    def shuffleDir(self):
        random.shuffle(self.Dir);
        
    def workMix(self):
        q = [];
        files = [];
        cnt = 0;
        PTS_NUM = None;
        PTS_DENSE_NUM = None;
        VIEW_NUM = None;
        while cnt < self.BATCH_SIZE:
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
                print datapath," contain invalid data in x2D";
            elif not np.isfinite(yGTIn).all():
                print datapath," contain invalid data in yGT";
            else:
                files.append((f,fdense));
                cnt += 1;
            self.Cnt += 1;
            if self.Cnt >= len(self.Dir):
                self.Cnt = 0;
                self.EpochCnt += 1;
        for i in range(VIEW_NUM):
            x2D = np.zeros([self.BATCH_SIZE,self.HEIGHT,self.WIDTH,4]);
            x3D = eval(self.randfunc);
            yGT = np.zeros([self.BATCH_SIZE,PTS_NUM,3]);
            if fdense is not None:
                yGTdense = np.zeros([self.BATCH_SIZE,PTS_DENSE_NUM,3]);
                q.append((x2D,x3D,yGT,yGTdense));
            else:
                q.append((x2D,x3D,yGT));
        fi = 0;
        for f,fdense in files:
            x2DIn = f["IMG"][...];
            yGTIn = f["PV"][...];
            yGTDense = None;
            if fdense is not None:
                yGTDense = fdense["PV"][...];
            for i in range(VIEW_NUM):
                q[i][0][fi,...] = x2DIn[i,...];
                q[i][2][fi,...] = yGTIn[i,...];
                if yGTDense is not None:
                    q[i][-1][fi,...] = yGTDense[i//2,...];
            f.close();
            fi += 1;
        return q;
    
    def workNoMix(self):
        q = [];
        tag = [];
        datapath = self.Dir[self.Cnt];
        f = h5py.File(datapath,"r");
        ftag = os.path.basename(datapath).split("_")[0];
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
        for i in range(num):
            x2D = np.zeros([self.BATCH_SIZE,self.HEIGHT,self.WIDTH,4]);
            x3D = eval(self.randfunc);
            yGT = np.zeros([self.BATCH_SIZE,PTS_NUM,3]);
            q.append((x2D,x3D,yGT));
            tag.append(ftag);
        for i in range(VIEW_NUM):
            qi = i // self.BATCH_SIZE ;
            qj = i % self.BATCH_SIZE;
            q[qi][0][qj,...] = x2DIn[i,...];
            q[qi][2][qj,...] = yGTIn[i,...];
        f.close();
        return q,tag;
    
    def run(self):
        while not self.stopped:
            if self.Dir is not None:
                q = [];
                tags = [];
                if self.useMix:
                    q = self.workMix();
                else:
                    q,tags = self.workNoMix();
                for v in q:
                    self.Data.put(v);
                for tag in tags:
                    self.DataTag.put(tag);
    
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
            