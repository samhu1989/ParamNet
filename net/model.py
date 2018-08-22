from psgn import *;
from atlas import *;

def build_model(name,settings={}):
    return eval(name)(settings);