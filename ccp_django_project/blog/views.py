# html 과 연결되는 python code 이다 

from django.shortcuts import render
from .models import Post
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.views.generic import (
    ListView,
    DetailView,
    CreateView,
    UpdateView,
    DeleteView
)
from .forms import ImageForm
from PIL import Image
import shutil
from .imageprocess import process
import os

from .color_detection import read_color, cropImage
# from django.http import HttpResponse

def home(request):
    context = {
        'posts' : Post.objects.all()
    }
    return render(request, 'blog/home.html', context)

def about(request):
    return render(request, 'blog/about.html', {'title': 'About'})

class PostListView(ListView):
    model = Post
    template_name = 'blog/home.html' #<app>/<model>_<viewtype>.html
    context_object_name = 'posts'
    ordering=['-date_posted']

class PostDetailView(DetailView):
    model = Post

class PostCreateView(LoginRequiredMixin, UserPassesTestMixin, CreateView):
    model = Post
    fields = ['title', 'content']

    def form_valid(self, form):
        form.instance.author=self.request.user
        return super().form_valid(form)

    def test_func(self):
        post=self.get_object()
        if self.request.user == post.author:
            return True
        return False
    

class PostUpdateView(LoginRequiredMixin, UpdateView):
    model = Post
    fields = ['title', 'content']

    def form_valid(self, form):
        form.instance.author=self.request.user
        return super().form_valid(form)

    def test_func(self):
        post=self.get_object()
        if self.request.user == post.author:
            return True
        return False

class PostDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Post
    success_url = '/'

    def test_func(self):
        post=self.get_object()
        if self.request.user == post.author:
            return True
        return False

class UploadView():
    model = ImageForm
    success_url = '/'

def upload(request):
    savePath='./media/input/uploaded.jpg'
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        # shutil.copyfile('.'+uploaded_file_url, savePath)
        os.rename('.'+uploaded_file_url, savePath)
        return render(request, 'blog/upload.html', {
            'uploaded_file_url': '/media/input/uploaded.jpg'
        })
    
    return render(request, 'blog/upload.html')

def extractName(fileName):
    with open(fileName, 'r') as file:
        # category_name=list(file)[-2].split(':')[0]
        labels=[label for label in list(file) if 'Label' in label]
        
        labelForShow=''
        for label in labels:
            currentLabelName=label.split(':')[1].split(',')[0]
            if currentLabelName not in labelForShow:
                labelForShow+=currentLabelName
            else:
                continue

    # return category_name
    return labelForShow

def ShowResult(request):

    clothesIndex = {
        0: "Blazer",
        1: "Blouse",
        2: "Button-Down",
        3: "Cardigan",
        4: "Coat",
        5: "Dress",
        6: "Hoodie",
        7: "Jacket",
        8:"Jeans",
        9:"Leggings",
        10:"Shorts",
        11:"Skirt",
        12:"Sweater",
        13:"Sweatpants",
        14:"Sweatshorts",
        15:"Tee",
        16:"Turtleneck"
    }

    command='python detect.py --image-folder ./media/input --output-folder ./media/output --txt-out True'
    os.system(command)

    resultIndexes = ExtractResultIndex(resultImageName ='uploaded.jpg',  outputFolderPath = './media/output')

    if len(resultIndexes) == 1:
        resultCordinates, resultIndex, resultProbability = resultIndexes[0]
        clothesName = clothesIndex[resultIndex]

        # TODO: Read color and export RGB color

        # color detection

        # read cropped image path
        resultImagePath = './media/uploaded.jpg'
        croppedImagePath = cropImage(resultImagePath, resultCordinates)

        # cloud 서비스이기 때문에 잠깐 꺼놓기!
        
        main_colors = read_color(croppedImagePath)
        # main_colors = (1,1,1)

        if len(main_colors) == 3:
            color_R = main_colors[0]
            color_G = main_colors[1]
            color_B = main_colors[2]

        else:

            color_R = "not detected"
            color_G = "not detected"
            color_B = "not detected"

        season='spring / summer'

        context = {
            'clothesName' : clothesName,
            'resultProbability' : resultProbability,
            'color_R' : color_R,
            'color_G' : color_G,
            'color_B' : color_B,
            'season' : season,
        }

        return render(request, 'blog/upload_result.html', context)

    elif len(resultIndexes) == 0:
        clothesName = "Not Detected"
        resultProbability = 0

        context = {
            'clothesName' : clothesName,
            'resultProbability' : resultProbability,
            'color' : " ",
            'season' : " "
        }

        return render(request, 'blog/upload_result.html', context)
    
    else:

        clothesName = "Too many Clothes"
        resultProbability = 0

        context = {
            'clothesName' : clothesName,
            'resultProbability' : resultProbability,
            'color' : ' ',
            'season' : ' '
        }
        return render(request, 'blog/upload_result.html', context)

def ExtractResultIndex(resultImageName, outputFolderPath):
    resultIndex = []
    textFilePath = resultImageName + '.txt' 
    with open(os.path.join(outputFolderPath, textFilePath)) as textFile:
        for line in textFile:
            resultIndex.append((line.split(' ')[:4], int(line.split(' ')[4]), float(line.split(' ')[5]) * 100 ))
            # resultIndex.append((line.split(' ')[:4], int(line.split(' ')[4]), line.split(' ')[5] ))
        
    return resultIndex