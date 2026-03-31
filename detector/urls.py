from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('demo/', views.demo, name='demo'),
    path('instructions/', views.instructions, name='instructions'),
    path('samples/', views.samples, name='samples'),
    path('about/', views.about, name='about'),
    path('api/process/', views.process_image_api, name='process_image_api'), 
]