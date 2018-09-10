from psgn import *;
from atlas import *;
from param import *;

def build_model(name,settings={}):
    return eval(name)(settings);