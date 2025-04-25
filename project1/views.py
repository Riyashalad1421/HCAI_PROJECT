from django.shortcuts import render
from django.http import HttpResponse


#Very basic view as an example
def index(request):
    return HttpResponse("Welcome to Project 1!")


# Create your views here.
