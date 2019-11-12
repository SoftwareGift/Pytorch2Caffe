#!/usr/bin/env python
# coding=utf-8

import numpy as numpy
import cv2,os,sys
import numpy as np

import caffe
# caffe.set_mode_gpu()
caffe.set_mode_cpu()
model_def = './PytorchToCaffe/mobilenet.prototxt'
model_weights = './PytorchToCaffe/mobilenet.caffemodel'

net = caffe.Net(model_def, model_weights, caffe.TEST)

IMAGE_SIZE = (200, 200)

transformer = caffe.io.Transformer({'blob1': net.blobs['blob1'].data.shape})
transformer.set_transpose('blob1', (2,0,1))       # (h,w,c)--->(c,h,w)
transformer.set_mean('blob1', numpy.array([104,117,123])) # BGR
transformer.set_raw_scale('blob1', 255)  # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('blob1', (2, 1, 0))  # RGB--->BGR

image = caffe.io.load_image('face.jpg')
transformed_image = transformer.preprocess('blob1', image)
net.blobs['blob1'].data[...] = transformed_image
output = net.forward()

import time 
for _ in range(20):
    output=net.forward()

tic=time.time()
for _ in range(10):
    output=net.forward()
avg=(time.time()-tic)/1000
fps=1/avg

print('avg: {} fps: {}'.format(avg,fps))
