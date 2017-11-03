from PIL import Image
import os

source_path = 'C:/Users/phuaj/Desktop/project/data/original'
dest_path = 'C:/Users/phuaj/Desktop/project/data/resize' 
paths = [source_path, dest_path]

#check directory exist
for p in paths:
	if not os.path.exists(p):
		print('Creating ''%s''' % p)
		os.makedirs(p)

print('source:', os.listdir(source_path))
print('destination:', os.listdir(dest_path))

#data augmentation
resize_width, reize_height = 128, 128
count = 0

def save_jpg(image, filename_ext):
	image.convert('RGB').save(im_dest_path + filename_ext + '.jpg', 'JPEG', quality=90)

for i in os.listdir(source_path):
	print('start to process:', i)
	im_source_path = source_path + '/' + str(i)
	im_dest_path = dest_path + '/' + str(count)
	im = Image.open(im_source_path)
	
	#resize
	im_flip = im_resize = im.resize((resize_width, reize_height), Image.ANTIALIAS)
	save_jpg(im_resize, '_resized')
	
	#flip
	im_resize.transpose(Image.FLIP_LEFT_RIGHT)
	save_jpg(im_resize, '_flip')
	
	#rotate
	size = [np.random.randint(150, 180), np.random.randint(170, 200), np.random.randint(190, 220), np.random.randint(210, 240)]
	random = [np.random.randint(0, 20), np.random.randint(20, 40), np.random.randint(-40, -20), np.random.randint(-20, 0)]
	box = [(j/2-resize_width/2, j/2-resize_width/2, j/2+resize_width/2, j/2+resize_width/2) for j in size]
	
	for j in range(len(size)):
		imRotate = im.rotate(random[j]).resize((size[j],size[j]), Image.ANTIALIAS).crop(box[j])
		imRotate2 = im_flip.rotate(random[j]).resize((size[j],size[j]), Image.ANTIALIAS).crop(box[j])
		save_jpg(imRotate, '_rotate')
		save_jpg(imRotate2, '_' + str(j+len(size)) + '_rotate')
	
	count += 1