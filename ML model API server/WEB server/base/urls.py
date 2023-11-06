from django.urls import path

from .views import index, image_stream, index2

urlpatterns = [
    path('', index, name='index'),
    path('index2', index2, name='index'),
    path('stream/<str:id_ref>', image_stream, name='image_stream'),
]