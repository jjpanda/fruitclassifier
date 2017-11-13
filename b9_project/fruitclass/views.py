from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.template import loader

from .forms import ImageForm

def index(request):
    print('index')
    if request.method == 'POST':
        test = request.FILES['img']
        print(type(test))
        print('predict_POST')
        form = ImageForm(request.POST, request.FILES)
        print('form')
    
    else:
        form = ImageForm()
        
    return render(
        request,
        'main/index.html',
        {'form': form}
    )
