# from django.http import HttpResponse


# def index(request):
#     return HttpResponse("Hello, world. You're at the polls index.")

from django.http import HttpResponse
from django.template import loader


def index(request):
    template = loader.get_template("home/index.html")
    
    
    students = [
        {"name": "Rita Barbosa", "matriculation": "645593"},
        {"name": "Yash Indulkar", "matriculation": "642352"},
        {"name": "Riyasha Lad", "matriculation": "641409"},
        {"name": "Shivangi Pathak", "matriculation": "641285"},
        {"name": "Vishwesh Jagtap", "matriculation": "641524"},
    ]
    
    projects = [
        {"name": "Home", "url_name": "home:index"},
        {"name": "Home 2", "url_name": "home:index"},
    ]
    
    context = { 
        "students": students, 
        "projects": projects, 
    }
    
    return HttpResponse(template.render(context, request))