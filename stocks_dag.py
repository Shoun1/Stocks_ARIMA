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
    #print(data[['Open','High','Low']].iloc[0:6])
    #print(data.tail())
    #print(data.shape)
    #print(data.info())
    #perform Exploratory Data Analysis
    #print(data.describe())
    data.to_csv('/home/shoun1/batch_data.csv',index=False)

def preprocess_data():
    data = pd.read_csv('/home/shoun1/batch_data.csv')
    data_melted = data.melt(value_vars=['Open', 'Low', 'High', 'Close'], 
                            var_name='Features', value_name='Values')

    # Plot with Seaborn
    plt.figure(figsize=(10, 6))
    #sns.boxplot(x='Features', y='Values', data=data_melted)
    #plt.title('Box Plot of Open, Low, High, and Close for Outlier Detection')
    #plt.savefig('/home/shoun1/airflow/dags/boxplot.png')
    

def train_model():
    data = pd.read_csv('/home/shoun1/batch_data.csv')
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
    return lm,x_train,y_train,x_test,y_test,data
    

def make_predictions(lm,x_train,y_train,x_test,y_test):
    #predicting closing price on test data model makes its own predictions
    #y_pred = lm.predict(x_train)
    y_pred = lm.predict(x_test)
    #predicting closing price on a new dataset model makes predictions on input data
    input_data = {'Open':[620,655,658,660,662,664],
            'High':[630,665,660,665,660,665],
            'Low':[620,625,630,640,645,650],
            'PrevClose':[630,635,640,645,650,655]}
    new_data = pd.DataFrame(input_data)

    new_data = StandardScaler().fit_transform(new_data)
    y_pred_data = lm.predict(new_data)

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

    df_compare = pd.DataFrame({'actual': y_test,'predicted': y_pred})
    print(df_compare)
    correlation = np.corrcoef(df_compare['actual'], df_compare['predicted'])
    print(f"Correlation: {correlation}")

    # Define accuracy as the % of predictions within ±₹10 of actual
    '''threshold = 10
    diff = np.abs(y_test - y_pred_data)
    accuracy = np.mean(diff <= threshold) * 100

    print(f"Custom Regression Accuracy (±₹{threshold}): {accuracy:.2f}%")'''


'''def process_predictions(x,x_train,y_train,x_test,y_test,y_pred):
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

def visualize_predictions():
    data = pd.read_csv('/home/shoun1/batch_data.csv')
    train,test = train_test_split(data,test_size=0.2)

    X_train = np.array(train.index).reshape(-1, 1)
    y_train = train['Close']
    # Create LinearRegression Object
    model = LinearRegression()
    # Fit linear model using the train data set
    model.fit(X_train, y_train)

    plt.figure(1, figsize=(16,10))
    plt.title('Linear Regression | Price vs Time')
    plt.scatter(X_train, y_train, edgecolor='w', label='Actual Price')
    plt.plot(X_train, model.predict(X_train), color='r', label='Predicted Price')
    plt.xlabel('Integer Date')
    plt.ylabel('Stock Price')
    plt.savefig('/home/shoun1/airflow/dags/Stocks_ARIMA/plots/linear_regression.png')
    

load_data(0,98)
preprocess_data()
lm,x_train,y_train,x_test,y_test,data = train_model()
make_predictions(lm,x_train,y_train,x_test,y_test)
visualize_predictions()
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

load_data_task >> preprocess_data_task >> train_model_task >> make_predictions_task >> process_predictions_task'''










