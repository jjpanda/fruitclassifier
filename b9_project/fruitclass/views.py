from django.shortcuts import render

# Create your views here.
from django.template import loader

from django.conf import settings
from django.forms.forms import Form
model = settings.MODEL

from .forms import ImageForm

from PIL import Image
from io import BytesIO

def index(request):
    print('index')
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            print(request.FILES)
            result = handle_uploaded_image(request.FILES['imagefile'])
            context  = {
                'text': result,
                'img' : request.FILES['imagefile'],
                'form': form
                }
            return render(request,'main/index.html', context )
    else:
        form = ImageForm()
        
    return render(request, 'main/index.html', {'form': form})

from fruitclass.util_00 import * 
from keras.models import load_model
import numpy as np


def _process_image(im):
    img_width, img_height = get_resize_dimension()
    im_resize = im.resize((img_width, img_height), Image.ANTIALIAS)
    im = np.array(im_resize)
    im = im.reshape(-1, img_width , img_height, 3)
    im = im.astype('float32')
    im /= 255
    return im

def predict_fruit(image):
    im = _process_image(image)
    #model = load_model('04_CNN.h5')
    result = model.predict(im)
    labels = get_tick_labels()
    print('results:', result)
    print('labels:', labels)
    d = dict(zip(labels, result[0]))
    print(d)
    s = [(k, d[k]) for k in sorted(d, key=d.get, reverse=True)]
    classes = np.argmax(result)
    print(s)
    labels = get_tick_labels()
    #return str(labels[classes])
    return s


def handle_uploaded_image(i):
    # resize image
    print('handle_uploaded_image')
    imagefile  = BytesIO(i.read())
    image = Image.open(imagefile)
    return predict_fruit(image)
