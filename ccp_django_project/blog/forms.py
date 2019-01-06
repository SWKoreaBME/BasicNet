from django import forms
# from uploads.core.models import Document
from .models import Yolo_image

class ImageForm(forms.ModelForm):
    class Meta:
        model = Yolo_image
        fields = ['image']