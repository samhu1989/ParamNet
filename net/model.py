from kparam33_dual import *;
from kparam33 import *;
from kparam34_dual import *;
from kparam35 import *;
from psgn import *;
from kparam42 import *;
from kparam43 import *;
from kparam44 import *;
from kparam45 import *;
from kparam46 import *;

def build_model(name,settings={}):
    return eval(name)(settings);