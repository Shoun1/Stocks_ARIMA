from django.urls import path
from django.http import HttpResponse
from . import views

urlpatterns=[
    path('predict_price',views.predict_price,name='predict_price'),
    path('home',views.home),
]