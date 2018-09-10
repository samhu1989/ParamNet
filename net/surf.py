from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

resnet_arg_scope = resnet_utils.resnet_arg_scope;

def kconv(W,X,b):
    return;

def duplicate_concate(x2d,x3d):
    return;

def duplicate_outerproduct(x2d,x3d):
    return;

def kconv_couple(x2d,x3d,param=[]):
    return;

def mlp(x2d,x3d,param=[]):
    return

def kconv_pts(x2d,x3d,param=[]):
    return;

def kconv_pts_res(x2d,x3d,param=[]):
    return;

def laplace(x2d,x3d,param=[]):
    return;

blocks={
    'C':duplicate_concate,
    'O',duplicate_outerproduct,
    'K',kconv_couple,
    'P',kconv_pts,
    'R',kconv_pts_res,
    'L',laplace
};

def surf_decoder(surf_dict={},net=None,grid_num,istrain,scope='surf_decoder',reuse=False):
    if net:
        sig = net[surf_dict['sig']];
        grid = net[surf_dict['grid']];
        surf = surf_dict['surf'];
        kidx = None;
        for b in surf:
            if len(b) == 2:
                x3d = blocks[b[0]](x2d,x3d,param=b[1]);
            if len(b) == 1:
                x3d = blocks[b[0]](x2d,x3d);
        net[surf_dict['output']] = x3d;
    