from django.urls import path
from . import views

app_name = 'project4'

urlpatterns = [
    path('', views.project4_landing, name='index'),
    path('start/', views.project4_study, name='study'),
    path('download/', views.project4_download_pdf, name='download'),
]