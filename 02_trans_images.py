import os

import numpy as np
from PIL import Image

import util_00 as util

def save_jpg(image, file_path):
	image.convert('RGB').save(file_path + '.jpg', 'JPEG', quality=90)
	
current_path = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_path, 'data', 'download', 'images')
dest_path = os.path.join(current_path, 'data', 'augmented')

#data augmentation
resize_width, resize_height = 128, 128

for i in os.listdir(src_path):
	print('Processing images in folder:', i)
	src_fruit_path = os.path.join(src_path, str(i)) #set to each fruit folder
	dest_fruit_path = os.path.join(dest_path, str(i)) #save image to respective fruit folder
	util.checkdir(dest_fruit_path)	
	for f_img in os.listdir(src_fruit_path):
		print('image:', f_img)
		count = 0
		im_source_path = os.path.join(src_fruit_path, str(f_img))
		try:
			im = Image.open(im_source_path)
			#destination folder
			im_name = os.path.basename(im_source_path) #get image name without file extension
			im_dest_path = os.path.join(dest_fruit_path,str(im_name) + '_' )
			
			#resize
			im_resize = im.resize((resize_width, resize_height), Image.ANTIALIAS)
			save_jpg(im_resize, im_dest_path + '_resized')
			
			#flip
			im_flip = im_resize.transpose(Image.FLIP_LEFT_RIGHT)
			save_jpg(im_flip, im_dest_path + '_flip')
			
			#rotate
			size = [np.random.randint(150, 180), np.random.randint(170, 200), np.random.randint(190, 220), np.random.randint(210, 240)]
			random = [np.random.randint(0, 20), np.random.randint(20, 40), np.random.randint(-40, -20), np.random.randint(-20, 0)]
			box = [(j/2-resize_width/2, j/2-resize_width/2, j/2+resize_width/2, j/2+resize_width/2) for j in size]
			
			for j in range(len(size)):
				imRotate = im.rotate(random[j]).resize((size[j],size[j]), Image.ANTIALIAS).crop(box[j])
				imRotate2 = im_flip.rotate(random[j]).resize((size[j],size[j]), Image.ANTIALIAS).crop(box[j])
				save_jpg(imRotate, im_dest_path + str(j) +  '_rotate')
				save_jpg(imRotate2, im_dest_path + '_' + str(j+len(size)) + '_rotate')
		except Exception as e:
			print('error image:', f)
			print(e)
