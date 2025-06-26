from django.shortcuts import render,redirect
from .models import *
from django.http import HttpResponse
from .prediction import load_data,train_model,make_predictions

data = load_data(0,98)
lm,x_train,y_train,x_test,y_test,scaler = train_model()


def home(request):
    return render(request,'home.html')

# Create your views here.
def predict_price(request):
    return render(request,'predict_price.html')

def predict_price(request):
    #price= Prices()
    y_pred_data = None
    if request.method == 'POST':
        '''price.Open = request.POST['Open']
        price.High = request.POST['High']
        price.PrevClose = request.POST['PrevClose']'''
        Open = request.POST['Open']
        High = request.POST['High']
        Low = request.POST['Low']
        PrevClose = request.POST['PrevClose']

        price = Prices.objects.create(
            Open=Open,
            High=High,
            Low=Low,
            PrevClose=PrevClose
        )

        price.save()

        try:
            if price is not None:
                y_pred_data = make_predictions(lm,x_train,y_train,x_test,y_test,scaler,price.Open,price.High,price.Low,price.PrevClose)
                #return render(request,'predict_price.html',{'predicted_price':y_pred})
                print(y_pred_data)

        except Exception as e:
            return HttpResponse("Price prediction failed: " + str(e), status=500)

    return render(request,'predict_price.html',{'y_pred_data':y_pred_data})


        