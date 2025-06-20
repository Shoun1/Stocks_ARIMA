from airflow import DAG
from datetime import datetime
import datetime as dt
from datetime import timedelta
from airflow.operators.python_operator import PythonVirtualenvOperator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

def load_data(lb,ub):
    df = pd.read_csv('/home/shoun1/lic.csv')
    data = df.iloc[lb:ub]
    #print(data.head())
    print(data[['Open','High','Low']].iloc[0:6])
    #print(data.tail())
    #print(data.shape)
    #print(data.info())
    #perform Exploratory Data Analysis
    #print(data.describe())
    data.to_csv('/home/shoun1/batch_data.csv',index=False)
    return data

def preprocess_data(data):
    #data = pd.read_csv('/home/shoun1/batch_data.csv')
    data_melted = data.melt(value_vars=['Open', 'Low', 'High', 'Close'], 
                            var_name='Features', value_name='Values')

    # Plot with Seaborn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Features', y='Values', data=data_melted)
    plt.title('Box Plot of Open, Low, High, and Close for Outlier Detection')
    plt.savefig('/home/shoun1/airflow/dags/plots/boxplot.png')
    

def train_model(data):
    #data = pd.read_csv('/home/shoun1/batch_data.csv')
    data.loc[:,'PrevClose'] = data['Close'].shift(1)

    data = data.dropna()
    #split dataframe into independent and dependent columns
    x = data[['Open','High','Low','PrevClose']]
    y = data['Close']
    #bifurcate training testing data
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=1,shuffle=False)
    #feature scaling
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    lm = LinearRegression()
    lm.fit(x_train,y_train)
    return lm,x_test,y_test
    

def make_predictions(data,lm,x_test,y_test):

    #predicting closing price on test data model makes its own predictions
    #y_pred = lm.predict(x_train)
    y_pred = lm.predict(x_test)
    
    #predicting closing price on a new dataset model makes predictions on input data
    input_data = {'Open':[620,655,658,660,662,664],
            'High':[630,665,660,665,660,665],
            'Low':[620,625,630,640,645,650],
            'PrevClose':[630,635,640,645,650,655]}

    new_data = pd.DataFrame(input_data)
    y_pred_data = lm.predict(new_data)

    X_train = data[
    (data['Open'].between(620, 660)) &
    (data['High'].between(630, 665)) &
    (data['Low'].between(620,650)) &
    (data['PrevClose'].between(630,655))
    ]

    X_train = X_train.drop(['Date','Close'],axis=1)


    # Sample similar rows
    y_actual = np.random.choice(y_pred.flatten(),size=4,replace=False)

    #convert to column vectors
    y_pred = np.array(y_pred).reshape(-1, 1)
    y_pred_data = np.array(y_pred_data).reshape(-1, 1)

    # Find closest values from y_pred to y_pred_data
    closest_values = []

    #data structure to hold the closest values
    for target in y_pred_data:
        distances = np.abs(y_pred - target)
        closest_index = np.argmin(distances)
        closest_values.append(y_pred[closest_index][0])

    Y_train = np.array(closest_values)

    print(X_train.shape)
    print(Y_train.shape)

    # 1️⃣ Pick ONE column from X_train
    x_vals = X_train['Open'].values          # shape: (4,)

# 2️⃣ Trim or sample Y_train so its length matches len(x_vals)
    y_vals = Y_train[:len(x_vals)]           # keep first 4 values

# 3️⃣ Now they are both 1‑D arrays of equal length
    plt.scatter(x_vals, y_vals,color='black')
    plt.xlabel('Open')
    plt.ylabel('Predicted Close (closest match)')
    plt.title('Open vs Predicted Close')
    plt.savefig('/home/shoun1/airflow/dags/Stocks_ARIMA/plots/scatter_plot.jpeg')




    # Then compare model predictions vs actual for those rows
    #X_similar = similar_rows.drop('Close', axis=1)
    #y_actual = similar_rows['Close']


    #y_pred = lm.predict(X_similar)
    #new_data = pd.DataFrame(input_data)
    #print(new_data)


    #X = new_data[['Open','High','Low','PrevClose']].values
    #Y = y_train[0:len(new_data)]
    #X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)
    #scaler = StandardScaler()
    #new_data_scaled = scaler.transform(new_data)

    #y_pred_data = lm.predict(new_data_scaled)



    '''y_pred_data = lm.predict(new_data)
    plt.scatter(Y_test,y_pred_data)
    plt.savefig('/home/shoun1/airflow/dags/newpred_com.jpeg')
    print("Predicted values: {}".format(y_pred_data))


    rmse = mean_squared_error(y_test,y_pred)
    print("Root mean squared error: {:.2f}".format(rmse))

    mae = mean_absolute_error(y_test,y_pred)
    print("Mean aboslute error: {:.2f}".format(mae))

    r2 = r2_score(y_test,y_pred)
    print("R2 score: {:.4f}".format(r2))
    #process_predictions(x,x_train,y_train,x_test,y_test,y_pred)

    m = lm.coef_[1]
    print("Regression coefficients: {}".format(m))
    c = lm.intercept_
    print("Regression intercept: {}".format(c))
    plt.scatter(y_test[0:10],y_pred[0:10])
    plt.savefig('/home/shoun1/airflow/dags/multiregr.jpeg')

    #print(len(x_train))
    #print(len(y_train))
    
    #plt.scatter(df['Open','High','Low','PrevClose'].values,df['Close'].values)
    #plt.scatter(x_train,y_train)
    #plt.plot(y_test,y_pred)

def process_predictions(x,x_train,y_train,x_test,y_test,y_pred):
    regr = LinearRegression()
    regr.fit(x_train,y_train)
    features = x.columns
    plt.figure(figsize=(10, 6))
    for idx, feature in enumerate(features, 1):
        plt.subplot(2, 2, idx)  # create a 2x2 grid for subplots
        plt.scatter(x_train[:, idx - 1], y_train, color='blue', label=f'{feature} vs Actual', alpha=0.6)
        plt.scatter(x_test[:, idx - 1], y_pred, color='red', label=f'{feature} vs Predicted', alpha=0.6)
        plt.xlabel(feature)
        plt.ylabel('Close')
        plt.legend()
        plt.title(f'Scatter Plot of {feature} vs Close')
        feature_range = np.linspace(x_test[:, idx - 1].min(), x_test[:, idx - 1].max(), 100).reshape(-1, 1)
        temp_x = np.zeros((100, x_train.shape[1]))  # fill a temporary array to hold other feature values constant
        temp_x[:, idx - 1] = feature_range[:, 0]
        #regression line
        line_pred = regr.predict(temp_x)
        
        plt.plot(feature_range, line_pred, color='black', label=f'Regression Line for {feature}')
        
        plt.xlabel(feature)
        plt.ylabel('Close')
        plt.legend()
        plt.title(f'Scatter Plot and Regression Line of {feature} vs Close')
        print("Regression coefficient: {}".format(regr.coef_))
        print("Regression intercept: {}".format(regr.intercept_))
        plt.savefig('/home/shoun1/airflow/dags/stocks_plot2.png')'''

data = load_data(0,98)
preprocess_data(data)
lm,x_test,y_test = train_model(data)
make_predictions(data,lm,x_test,y_test)

'''default_args = {
    'owner':'shoun10',
    'start_date' : dt.datetime(2023,10,20),
    'retries' : 1,
    'retry_delay': dt.timedelta(minutes=5),
}

with DAG(
    'stockprice_tracker',
    default_args = default_args,
    schedule_interval = '@daily',
    catchup = False
) as dag:

    load_data_task = PythonVirtualenvOperator(
        task_id = 'load_data',
        python_callable=load_data,
        op_args=[0,98],
        dag=dag
    )

    preprocess_data_task = PythonVirtualenvOperator(
        task_id = 'preprocess_data',
        python_callable = preprocess_data,
        dag = dag
    )

    train_model_task = PythonVirtualenvOperator(
        task_id = 'train_model',
        python_callable = train_model,
        dag=dag
    )

    make_predictions_task = PythonVirtualenvOperator(
        task_id = 'make_predictions',
        python_callable=make_predictions,
        op_args=[x,x_train,y_train,x_test,y_test],
        dag=dag
    )

    process_predictions_task = PythonVirtualenvOperator(
        task_id = 'process_predictions',
        python_callable= process_predictions,
        op_args=[x,x_train,y_train,x_test,y_test,y_pred],
        provide_context = True,
        dag=dag
    )

load_data_task >> preprocess_data_task >> train_model_task''' 










