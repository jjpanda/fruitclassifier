from django.db import models

# Create your models here.
class ImageModel(models.Model):
    docfile = models.ImageField("Avatar", blank=False, null=True)