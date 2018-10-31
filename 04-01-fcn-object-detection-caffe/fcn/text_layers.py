import caffe

import numpy as np
from PIL import Image
import cv2
import random
from data_agu import DataAugmentation

class TextSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - voc_dir: path to PASCAL VOC year dir
        - split: train / val / test
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)
	
        for PASCAL VOC semantic segmentation.

        example

        params = dict(voc_dir="/path/to/PASCAL/VOC2011",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="val")
        """
        # config
        params = eval(self.param_str)
        self.voc_dir = params['voc_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.voc_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
	tmp = self.indices[self.idx].split()
        self.data = self.load_image(tmp[0])
        self.label = self.load_label(tmp[1])
	
	#print "data = %s " % tmp[0]
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        - data_agumetation(add gaussian noise and color jittering) (add by Peter Huang)
        """
        im = Image.open(idx)  # Image.open('{}/JPEGImages/{}.jpg'.format(self.voc_dir, idx))
       	im = im.resize((480, 480),Image.ANTIALIAS)  
	image_in = im
	if False:#'val' in self.split:
		#print " val phase"
		cv_rgb = np.array(image_in, dtype=np.float32)
		hsv = cv2.cvtColor(cv_rgb,cv2.COLOR_RGB2HSV)
		hsv[:,:,2]*=0.6
		cv_rgb_changed = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
		cv_rgb_changed = np.array(cv_rgb_changed, dtype=np.uint8)
		image_in = Image.fromarray(cv_rgb_changed)

	in_ = np.array(image_in, dtype=np.float32)
        #in_ = in_[:,:,::-1]
        #in_ -= self.mean
	gray_image = cv2.cvtColor(in_, cv2.COLOR_RGB2GRAY)
	#gray_image = (gray_image - np.mean(gray_image))/(np.std(gray_image)+0.0001)
	gray_image = gray_image[np.newaxis, ...]
	#print gray_image.shape
        #gray_image = gray_image.transpose((2,0,1))
	#print gray_image.shape
	return gray_image


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
	#print idx
        im = Image.open(idx) #Image.open('{}/SegmentationClass/{}.jpg'.format(self.voc_dir, idx))
       	im = im.resize((480, 480),Image.ANTIALIAS)
	
	object_list = ['folgers','crayola','kleenex','viva','vanish','milo','swissmiss','cocacola','raisins','mm','andes','pocky','kotex','macadamia','stax','kellogg','hunts','3m','heineken','libava']
	### temporary change ###

	try:

		for obj_idx in range(0, len(object_list)):


			if idx.find(object_list[obj_idx])!=-1 :
				im_mask = np.asarray(im)
				area = im_mask[:,:] != 0
				seg = np.zeros((480,480), dtype=np.uint8)
				seg[area==True] = obj_idx + 1
				#print seg[seg!=0]
				break




	except:
		
		
		for obj_idx in range(0, len(object_list)):


			if idx.find(object_list[obj_idx])!=-1 :
				im_mask = np.asarray(im)
				area = im_mask[:,:,0] != 0
				seg = np.zeros((480,480), dtype=np.uint8)
				seg[area==True] = obj_idx + 1
				#print seg[seg!=0]
				break



	label = np.array(seg, dtype=np.uint8)

	### temporary change ###
	
        #label = np.array(im, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label
