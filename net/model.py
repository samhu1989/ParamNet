from kparam33_dual import *;
from kparam33 import *;
from kparam34_dual import *;

def build_model(name,settings={}):
    return eval(name)(settings);