import numpy as np;
from PIL import Image;
import os;
import tensorflow as tf;
from tensorflow.python import pywrap_tensorflow;

def listdir(dir_,suffix=None):
    lst = os.listdir(dir_);
    olst = [];
    for i in range(len(lst)):
        if suffix is None:
            olst.append( dir_+os.sep+lst[i] );
        elif lst[i].endswith(suffix):
            olst.append( dir_+os.sep+lst[i] );
    return olst;

def write_to_obj(fpath,pts_v,pts_c=None,faces=None):
    for i in range(pts_v.shape[0]):
        fpath_i = fpath+"_%02d.obj"%i;
        f = open(fpath_i,"w");
        for j in range(pts_v.shape[1]):
            if pts_c is not None and pts_v.shape == pts_c.shape:
                print >>f,"v %f %f %f %f %f %f"%(pts_v[i,j,0],pts_v[i,j,1],pts_v[i,j,2],pts_c[i,j,0],pts_c[i,j,1],pts_c[i,j,2]);
            else:
                print >>f,"v %f %f %f"%(pts_v[i,j,0],pts_v[i,j,1],pts_v[i,j,2]);
        if faces is not None and len(faces)==pts_v.shape[0]:
            face = faces[i];
            for pi in range(face.shape[0]):
                print >>f,"f %d %d %d"%(face[pi,0]+1,face[pi,1]+1,face[pi,2]+1);
        f.close();
    return;

def write_to_ply(fpath,pts_v,pts_n=None,faces=None):
    return;
    

def write_to_img(path,img):
    N = img.shape[0];
    for n in range(N):
        imn = img[n,:,:,:];
        imn *= 255.0;
        im = Image.fromarray(imn.astype(np.uint8));
        im.save(path+os.sep+"img%d.png"%n);
        
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