from keras.models import load_model
from PIL import Image
import numpy as np

import util_00 as util

def _process_image(im):
    img_width, img_height = util.get_resize_dimension()
    im_resize = im.resize((img_width, img_height), Image.ANTIALIAS)
    im = np.array(im_resize)
    im = im.reshape(-1, img_width , img_height, 3)
    im = im.astype('float32')
    im /= 255
    return im

def predict_fruit(image):
    im = _process_image(image)
    result = model.predict(im)
    classes = np.argmax(result)
    labels = util.get_tick_labels()
    return str(labels[classes])

model = load_model('04_CNN.h5')
im1 = Image.open('orange.jpg')
im2 = Image.open('pineapple.jpg')
im3 = Image.open('apple.jpg')

test_images = [im1, im2, im3]

for t in test_images:
    print(predict_fruit(t))