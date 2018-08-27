import numpy as np;
import h5py;
import random;
import os;
from PIL import Image,ImageOps;
import io;
import sampling;

def genRealBatch(imgpath):
    lst = io.listdir(imgpath,'.jpg');
    for fname in lst:
        print fname;
        mskname = os.path.dirname(fname) + os.sep + os.path.basename(fname).split('.')[0];
        im_array = None;
        msk_array = None;
        if os.path.exists(mskname+".png"):
            im = Image.open(fname);
            msk = ImageOps.grayscale(Image.open(mskname+".png"));
            im_array = np.asarray(im);
            msk_array = np.asarray(msk);
        elif os.path.exists(mskname+".PNG"):
            im = Image.open(fname);
            msk = ImageOps.grayscale(Image.open(mskname+".PNG"));
            im_array = np.asarray(im);
            msk_array = np.asarray(msk);
        if im_array is not None and msk_array is not None:
            print im_array.shape;
            print msk_array.shape;
        else:
            continue;
        pv_data = sampling.rand_n_sphere(32,1024);
        im_data = np.zeros([32,192,256,4],dtype=np.float32);
        
        imo_array = im_array.copy();
        alpha_array = imo_array[:,:,0].copy();
        alpha_array.fill(0.0);
        alpha_array[msk_array > 128] = 255;
        imo_array[msk_array < 128, : ] = 128;
        imo = Image.fromarray(imo_array);
        alphao = Image.fromarray(alpha_array);
        imo.putalpha(alphao);
        box = alphao.getbbox();
        imocrop = imo.crop(box);
        sx = float(box[2] - box[0]);
        sy = float(box[3] - box[1]);
        stxo = 128.0;
        styo = 96.0;
        slst = np.linspace(1.0,1.2,8,dtype=np.float);
        dlst = np.linspace(-5,5,4,dtype=np.int);
        for i,s in enumerate(slst):
            stx = stxo*s;
            sty = styo*s;
            print stx,sty
            rx = stx / sx;
            ry = sty / sy;
            if rx > ry:
                stx_adj = int(ry * sx) ;
                sty_adj = int(sty);
            elif rx < ry:
                stx_adj = int(stx);
                sty_adj = int(rx * sy) ;
            for j,d in enumerate(dlst): 
                cx = 128 - stx_adj//2 + d;
                cy = 96  - sty_adj//2 + d;
                print (i,j);
                resize_imo = imocrop.resize( size=(stx_adj,sty_adj) , resample=Image.BICUBIC);
                imoj = Image.new("RGBA",(256,192));
                imoj.paste(resize_imo,(cx,cy));
                im_data[i*4+j,:,:,:] = np.asarray(imoj).astype(np.float32).copy();
        im_data /= 255.0;
        f = h5py.File(mskname+".h5","w");
        f.create_dataset('IMG', data=im_data,compression="gzip", compression_opts=9);
        f.create_dataset('PV',data=pv_data['x3D'],compression="gzip", compression_opts=9);
        f.close();
            
        
    
    
    
    
    
    
    
    
    