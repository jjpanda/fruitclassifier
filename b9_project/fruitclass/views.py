from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.template import loader

from .forms import ImageForm

def index(request):
    print('index')
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_image(request.FILES['img'])
            #return HttpResponseRedirect(elsewhere)
    else:
        form = ImageForm()
        
    return render(
        request,
        'main/index.html',
        {'form': form}
    )

def handle_uploaded_image(i):
    # resize image
    imagefile  = StringIO.StringIO(i.read())
    imageImage = Image.open(imagefile)

    (width, height) = imageImage.size
    (width, height) = scale_dimensions(width, height, longest_side=240)

    resizedImage = imageImage.resize((width, height))

    imagefile = StringIO.StringIO()
    resizedImage.save(imagefile,'JPEG')
    filename = hashlib.md5(imagefile.getvalue()).hexdigest()+'.jpg'

    # #save to disk
    imagefile = open(os.path.join('/tmp',filename), 'w')
    resizedImage.save(imagefile,'JPEG')
    imagefile = open(os.path.join('/tmp',filename), 'r')
    content = django.core.files.File(imagefile)

    my_object = MyDjangoObject()
    my_object.photo.save(filename, content)
