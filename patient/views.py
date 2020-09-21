from django.shortcuts import render
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from .models import Test
from django.views.generic import (
    ListView,
    DetailView,
    CreateView,
    UpdateView,
    DeleteView
)
def preProcess(request):
    return render(request, 'patient/preProcess.html', context)
    context = {'a': 1}

def showPreprocessImages(request):
    context = {'a': 1}
    return render(request, 'patient/preProcess.html', context)

def prepare(filepath):
    IMG_SIZE = 60
    #80 for shivani_bigdataset 60 for 900 dataset model and 30 100 dataset
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return img_array.reshape(-1,IMG_SIZE,IMG_SIZE ,1)


def home(request):
    context = {
    # only user's tests here
        'tests': Test.objects.all()
    }
    return render(request, 'patient/home.html', context)

def about(request):
    return render(request, 'patient/about.html', {'title': 'About'})

class TestListView(ListView):
    model = Test
    template_name = 'patient/home.html'  # <app>/<model>_<viewtype>.html
    context_object_name = 'tests'
    ordering = ['-date_tested']

class TestDetailView(DetailView):
    model = Test

class TestCreateView(LoginRequiredMixin, CreateView):
    model = Test
    fields = ['test_type', 'description', 'eye_image', 'test_result']

    def form_valid(self, form):
        form.instance.p_user = self.request.user
        return super().form_valid(form)

class TestUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Test
    fields = ['test_type', 'description', 'eye_image', 'test_result']

    def form_valid(self, form):
        form.instance.p_user = self.request.user
        return super().form_valid(form)

    def test_func(self):
        test = self.get_object()
        if self.request.user == test.p_user:
            return True
        return False

class TestDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Test
    success_url = '/'

    def test_func(self):
        post = self.get_object()
        if self.request.user == post.p_user:
            return True
        return False


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
from tensorflow.keras.callbacks import TensorBoard
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def output(request):
    img=Test.objects.filter(p_user=request.user)
    print('username : ' , request.user.username)
    print('img : ' , img)
    current_img = img[0].eye_image.name
    for i in img:
        print('img in for : ', i.eye_image.name)
        current_img = i.eye_image.name
    model = tf.keras.models.load_model("C:/Users/Lenovo/Desktop/D_BE_project/projectmodule/dr-cnn-900.model")
    url='C:/Users/Lenovo/Desktop/D_BE_project/Django/diabetese_retinopathy_project/media/'+current_img
    print('url : ', url)
    print('url llast : ',Test.objects.last().eye_image.name)
    i = prepare(url)
    i = tf.cast(i,tf.float32)
    prediction = model.predict(i)
    print(prediction)
    o = prediction[0][0]
    print('o', o)
    map={0:'negative',1:'positive'}
    print("prediction: - ",map[o])
    result = map[o]
    return render(request, 'patient/output.html', {'result': result})

def leveltest(request):

    result = "--"
    return render(request, 'patient/test_level.html', {'result': result})

def preprocess(request):
    print("hellloe" , request )
    result = "Image"
    return render(request, 'patient/preprocess.html', {'result': result})
