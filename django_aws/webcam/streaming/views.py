from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
from django.shortcuts import redirect, render




def index(request):
    return render(request, 'page1/page1.html')