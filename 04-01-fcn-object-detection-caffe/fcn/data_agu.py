from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import random
import os

root_Directory = os.getcwd() # current path

class DataAugmentation:

	def __init__(self):
		pass
	@staticmethod  
	def openImage(image):
		try:
			img = Image.open(image, mode="r")
        		return img
		except:
			return None
	@staticmethod  
	def randomColor(image):
		'''
		color jittering
		'''

		random_factor = np.random.randint(8, 13) / 10.  #
		color_image = ImageEnhance.Color(image).enhance(random_factor)  # image saturation
		random_factor = np.random.randint(8, 13) / 10.  
		brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # image brightness
		random_factor = np.random.randint(8, 13) / 10.  
		contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # image comaprision
		random_factor = np.random.randint(8, 13) / 10. 
		return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # sharpen


	@staticmethod  
	def randomGaussian(image, mean=0.2, sigma=0.3):
		'''
		Gaussian noise
		'''

		def gaussianNoisy(im, mean=0.2, sigma=0.3):
		    """
		    param im
		    param mean
		    param sigma

		    """
		    #for _i in range(len(im)):
		        #im[_i] += random.gauss(mean, sigma)
		    s = np.random.normal(loc=0, scale=1, size=len(im))
		    float_im = im.astype(np.float64)
		    float_im += s
		    return np.uint8(float_im)


		img = np.asarray(image)
		img.flags.writeable = True  

		width, height = img.shape[:2]
		img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
		img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
		img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
		img[:, :, 0] = img_r.reshape([width, height])
		img[:, :, 1] = img_g.reshape([width, height])
		img[:, :, 2] = img_b.reshape([width, height])
		return Image.fromarray(np.uint8(img))

	@staticmethod  
	def saveImage(image, path):
        	image.save(path)


def main():
	object_list = ['folgers','crayola','kleenex','viva','vanish','milo','swissmiss','cocacola','raisins','mm','andes','pocky','kotex','macadamia','stax','kellogg','hunts','3m','heineken','libava']
	pic_path = root_Directory + '/img'
	agu_path = root_Directory + '/agument_full'
	
	for obj in range(0,20):
		for i in range(0,46):
			for j in range(0,31):
				sub_path = "/%s/scene_%06d/%06d.jpg" %(object_list[obj], i, j)
				
				original_path = pic_path + sub_path
				agu_path_full = agu_path + sub_path

				print original_path
				original = DataAugmentation.openImage(original_path)

				if original is  None:
					continue

				sharp = DataAugmentation.randomColor(original)
				noise = DataAugmentation.randomGaussian(sharp)
				
				object_path  = (agu_path +  "/%s" %(object_list[obj]))
				if not(os.path.isdir(object_path)):
					os.mkdir(object_path)

				tmp = (agu_path +  "/%s/scene_%06d" %(object_list[obj], i))
				if not(os.path.isdir(tmp)):
					os.mkdir(tmp)
				
				DataAugmentation.saveImage(noise,agu_path_full)




if __name__ == "__main__":
	main()