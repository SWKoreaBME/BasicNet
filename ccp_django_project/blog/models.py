from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from django.urls import reverse
from PIL import Image

class Post(models.Model):
    title = models.CharField(max_length=100)
    # image = models.ImageField(upload_to='pic_folder/', default='pic_folder/None/no-img.jpg')
    content = models.TextField() # 무제한적
    date_posted = models.DateTimeField(default=timezone.now)
    author = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.title
    
    def get_absolute_url(self):
        return reverse("model_detail", kwargs={"pk": self.pk})
    
class Document(models.Model):
    description = models.CharField(max_length=255, blank=True)
    document = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class Yolo_image(models.Model): 
    image = models.ImageField(upload_to='yolo_pics')

    def __str__(self):
        return f'{self.image.name} Profile'

    def save(self):
        super().save()

        img = Image.open(self.image.path)

        if img.height > 400 or img.width > 400:
            output_size = (400, 400)
            img.thumbnail(output_size)
            img.save(self.image.path)