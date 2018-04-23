# -*- coding: utf-8 -*-
import os;
import tensorflow as tf;
nvcc = "nvcc";
cxx = "g++";
cudalib = "/usr/local/cuda-8.0/lib64/";
TF_INC = tf.sysconfig.get_include();
TF_LIB = tf.sysconfig.get_lib();
eigen = "../../3rdParty/Eigen3"

os.system(nvcc+" -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o ./ChamferDist.cu.o ./ChamferDist.cu -I "+TF_INC+" -I "+TF_INC+"/external/nsync/public"+" -L"+TF_LIB+" -ltensorflow_framework"+" -I "+eigen+" -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2");
os.system(cxx+" -std=c++11 ./ChamferDist.cpp ./ChamferDist.cu.o -o ./ChamferDist.so -shared -fPIC -I "+TF_INC+" -I "+TF_INC+"/external/nsync/public"+" -L"+TF_LIB+" -ltensorflow_framework"+" -I "+eigen+" -DGOOGLE_CUDA=1 -lcudart -L "+cudalib+" -O2 -D_GLIBCXX_USE_CXX11_ABI=0");
