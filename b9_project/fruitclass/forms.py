from django import forms

class ImageForm(forms.Form):
    print('image')
    imagefile = forms.ImageField(label='Photo', required=True)