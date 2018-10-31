from __future__ import division
import caffe
import numpy as np
import os
import sys
from datetime import datetime
from PIL import Image
#from execel_module import output_xls

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, save_dir, dataset, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    if save_dir:
        os.mkdir(save_dir)
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    for idx in dataset:
	#print idx
        net.forward()
        hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
                                net.blobs[layer].data[0].argmax(0).flatten(),
                                n_cl)

        if save_dir:
	    im = Image.fromarray(net.blobs[layer].data[0].argmax(0).astype(np.uint8), mode='P')
	    tmp = idx[1]
	    im.save(os.path.join(save_dir, tmp.replace('/','_') ))#+ '.png'))
	    print 'save as: ' + os.path.join(save_dir, tmp.replace('/','_') )#+ '.png')

        # compute the loss as well
        loss += net.blobs['loss'].data.flat[0]
    return hist, loss / len(dataset)

def seg_tests(solver, save_format, dataset, model_name, layer='score', gt='label'):
    print '>>>', datetime.now(), 'Begin seg tests'
    solver.test_nets[0].share_with(solver.net)
    do_seg_tests(solver.test_nets[0], solver.iter, save_format, dataset, model_name, layer, gt)

def do_seg_tests(net, iter, save_format, dataset, model_name, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    if save_format:
        save_format = save_format.format(iter)
    hist, loss = compute_hist(net, save_format, dataset, layer, gt)

    
    f = open("model/tests_log.txt","a+")
    #f1 = open("loss_log.txt","a+")
    #f2 = open("mean_acc_log.txt","a+")
    #f3 = open("mean_iou_log.txt","a+")
    #f4 = open("each_cls_acc_log.txt","a+")
    #f5 = open("each_cls_iou_log.txt","a+")

    f6 = open("model/mean.txt","a+")
    f7 = open("model/each_acc.txt","a+")
    f8 = open("model/each_iou_log.txt","a+")
    f9 = open("model/each_precision.txt","a+")
    f10 = open("model/each_recall.txt","a+")
    f11 = open("model/each_f_score.txt","a+")













    # mean loss
    print '>>>', datetime.now(), 'Iteration', iter, 'loss', loss
    f.write ('iter: ' + str(iter))
    f.write ('\n')
    f.write ('loss: ' + str(loss) + '\n' )
    #f1.write (str(loss) + '\n' )
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc
    f.write('overall accuracy: ' + `acc` + '\n')
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print '>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(acc)
    f.write('mean accuracy: ' + `np.nanmean(acc)` + '\n' )
    #f2.write(`np.nanmean(acc)` + '\n' )
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu)
    f.write('mean IU: ' + `np.nanmean(iu)` + '\n')
    #f3.write(`np.nanmean(iu)` + '\n')
    freq = hist.sum(1) / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc', \
            (freq[freq > 0] * iu[freq > 0]).sum()
    f.write('fwavacc: ' + `(freq[freq > 0] * iu[freq > 0]).sum()` + '\n')
    # each class accuracy
    print 'each class accuracy: ' + `acc`
    f.write('each class accuracy: ' + np.array2string(acc,max_line_width=1024) + '\n')
    
    shape = acc.shape
    #for i in range (0,shape[0] ):
      #f4.write(str(acc[i])+ ' ')
    #f4.write('\n')

    # each class IU
    print 'each class IU: ' + `iu`
    f.write('each class IU: ' + np.array2string(iu,max_line_width=1024) + '\n')

    #for i in range (0,shape[0]):
      #f5.write(str(iu[i])+ ' ')
    #f5.write('\n')
    


    each_cls_precision = np.diag(hist)/ hist.sum(0)
    each_cls_recall = np.diag(hist)/ hist.sum(1)
    each_F_score = 2*each_cls_precision*each_cls_recall/(each_cls_recall+each_cls_precision)

    mean_precision = np.nanmean(each_cls_precision)
    mean_recall = np.nanmean(each_cls_recall)
    mean_f_score = np.nanmean(each_F_score)


    # xls: iter, loss, mean_acc, mean_iou, mean_precision, mean_recall,

    xls_mean = [str(iter), str(loss), str(np.nanmean(acc)), str(np.nanmean(iu)), str(mean_precision), str(mean_recall), str(mean_f_score)]
    shape = len(xls_mean)
    for str_ in xls_mean:
      f6.write(str_)
      f6.write(' ')
    f6.write('\n')

    # each_acc
    xls_each_acc = np.array2string(acc,max_line_width=1024)

    shape = len(xls_each_acc)
    for str_ in xls_each_acc:
      f7.write(str_)
    f7.write('\n')
    # each_iou
    xls_each_iou = np.array2string(iu,max_line_width=1024)

    shape = len(xls_each_iou)
    for str_ in xls_each_iou:
      f8.write(str_)
    f8.write('\n')

    # each_precision
    xls_each_precision = np.array2string(each_cls_precision,max_line_width=1024)

    shape = len(xls_each_precision)
    for str_ in xls_each_precision:
      f9.write(str_)
    f9.write('\n')
    # each recall
    xls_each_recall = np.array2string(each_cls_recall,max_line_width=1024)

    shape = len(xls_each_recall)
    for str_ in xls_each_recall:
      f10.write(str_)
    f10.write('\n')
    # each f-score
    xls_f_score = np.array2string(each_F_score,max_line_width=1024)

    shape = len(xls_f_score)
    for str_ in xls_f_score:
      f11.write(str_)
    f11.write('\n')

    f.close()
    #f1.close()
    #f2.close()
    #f3.close()
    #f4.close()
    #f5.close()
    f6.close()
    f7.close()
    f8.close()
    f9.close()
    f10.close()
    f11.close()
    print hist
    return hist
