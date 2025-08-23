from django.urls import path
from . import views

app_name = 'project5'

urlpatterns = [
    path('', views.project5_landing, name='index'),
    path('test/', views.test_environment_direct, name='test'),
    path('test-models/', views.test_models, name='test_models'),
]