from kparam33_dual import *;

def build_model(name,settings={}):
    return eval(name)(settings);