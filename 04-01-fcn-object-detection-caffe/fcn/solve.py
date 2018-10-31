import sys
sys.path.append('/opt/caffe/python')
import caffe
import surgery, score

import numpy as np
import os
import sys
import argparse

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass
#Argument------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--weight', type=str, help='Input the path of the desired weight of pretrain model, default is vgg_dictnet_fcn_init_model.caffemodel', dest="weights", default='VGG-Dictnet-16s/vgg_dictnet_fcn_init_model.caffemodel')
parser.add_argument('--prototxt', type=str, help='Input the path of desired solver.prototxt, default is VGG-Dictnet-16s/solver.prototxt', dest="prototxt", default='VGG-Dictnet-16s/solver.prototxt')
#--------------------------------

args = parser.parse_args()

weights = args.weights

# init
#caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

model_name = args.prototxt 

solver = caffe.SGDSolver(args.prototxt)
solver.net.copy_from(weights)
#solver.restore('/media/peter/Blue_Others/VGG_Dictnet_model/snapshot/train_iter_200000.solverstate')

model_name = model_name.replace("/solver.prototxt", "")

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

if model_name == "VGG-Dictnet-16s": 
    val_name = "dataset-brandname_20_products/val.txt" 
else:
    val_name = "dataset-object_20_products/val.txt"
# scoring
val = np.loadtxt( val_name, dtype=str)

for _ in range(1):
    solver.step(4000)
    score.seg_tests(solver, False, val, model_name, layer='score')
