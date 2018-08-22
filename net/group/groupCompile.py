# -*- coding: utf-8 -*-
import os;
import tensorflow as tf;
nvcc = "/usr/local/cuda-8.0/bin/nvcc";
cxx = "g++";
cudalib = "/usr/local/cuda-8.0/lib64/";
TF_INC = tf.sysconfig.get_include();
TF_LIB = tf.sysconfig.get_lib();

os.system(nvcc+" -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o ./group.cu.o ./group.cu -I "+TF_INC+" -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2");
os.system(cxx+" -std=c++11 ./group.cpp ./group.cu.o -o ./group.so -shared -fPIC -I "+TF_INC+" -lcudart -L "+cudalib+" -O2 -D_GLIBCXX_USE_CXX11_ABI=0");

