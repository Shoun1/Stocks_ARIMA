from django.shortcuts import render,redirect
from .models import *
from django.http import HttpResponse


# Create your views here.
def predict_price(request):
    return render(request,'predict_price.html')

def predict_price(request):
    price= Prices()
    if request.method == 'POST':
        price.Open = request.POST['Open']
        price.High = request.POST['High']
        price.Close = request.POST['Close']
        price.PrevClose = request.POST['PrevClose']

    '''stock_price = Prices.objects.create(
        Open=Open,
        High=High,
        Close=Close,
        PrevClose=PrevClose
    )

    stock_price.save()
    try:
        if stock_price is not None:
            print(type(stock_price))
            predictions = {'predicted_price':0}

        return render(request, 'predict_price.html', {'predictions': predictions})

    except Exception as e:
            return HttpResponse("Error creating stock record.")'''

    return render(request,'predict_price.html')