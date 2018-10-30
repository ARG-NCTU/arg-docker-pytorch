import sys
from surgery import transplant
caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')


import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
import os
caffe.set_mode_gpu()

old_model_def = '/home/peter/caffe/fcn.berkeleyvision.org/32s_16s_8s_fcn_training/vgg_16s/deploy.prototxt'
old_model_weights = '/home/peter/caffe/fcn.berkeleyvision.org/32s_16s_8s_fcn_training/vgg_16s/snapshot/solver_iter_100000.caffemodel'

old_net = caffe.Net(old_model_def,      # defines the structure of the model
                old_model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


new_model_def = '/home/peter/caffe/fcn.berkeleyvision.org/32s_16s_8s_fcn_training/vgg_8s/deploy.prototxt'


new_net = caffe.Net(new_model_def,      # defines the structure of the model
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


transplant(new_net, old_net, suffix='')

new_net.save('vgg_16_8s_init.caffemodel')
