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
    '''print(data.head())
    print(data[['Open','High','Low']].iloc[0:6])
    print(data.tail())
    print(data.shape)
    print(data.info())
    #perform Exploratory Data Analysis
    print(data.describe())'''
    data.to_csv('/home/shoun1/batch_data.csv',index=False)

def preprocess_data():
    data = pd.read_csv('/home/shoun1/batch_data.csv')
    data_melted = data.melt(value_vars=['Open', 'Low', 'High', 'Close'], 
                            var_name='Features', value_name='Values')

    # Plot the box plot for outlier detection
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Features', y='Values', data=data_melted)
    plt.title('Box Plot of Open, Low, High, and Close for Outlier Detection')
    plt.savefig('/home/shoun1/airflow/dags/Stocks_ARIMA/plots/boxplot.png')

    # Plotting the distribution of each feature
    for feature in ['Open', 'High', 'Low', 'Close']:
        plt.figure(figsize=(8, 5))
        sns.kdeplot(data['Open'],fill=True,color='skyblue',linewidth=2)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.savefig(f'/home/shoun1/airflow/dags/Stocks_ARIMA/plots/{feature}_distribution.png')
    

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
    

def make_predictions(lm,x_train,y_train,x_test,y_test,data):
    #predicting closing price on test data model makes its own predictions
    #y_pred = lm.predict(x_train)
    y_pred = lm.predict(x_test)
    
    np.random.seed(42)
    input_data = {
        'Open': np.random.randint(500, 801, 20).tolist(),
        'High': np.random.randint(500, 801, 20).tolist(),
        'Low': np.random.randint(500, 801, 20).tolist(),
        'PrevClose': np.random.randint(500, 801, 20).tolist()
    }

    new_data = pd.DataFrame(input_data)

    new_data = StandardScaler().fit_transform(new_data)
    y_pred_data = lm.predict(new_data)

    min_open, max_open = data['Open'].min(), data['Open'].max()
    min_high, max_high = data['High'].min(), data['High'].max()
    min_low, max_low = data['Low'].min(), data['Low'].max()
    min_prevclose, max_prevclose = data['PrevClose'].min(), data['PrevClose'].max()

    filtered_data = data[
        (data['Open'] >= min_open) & (data['Open'] <= max_open) &
        (data['High'] >= min_high) & (data['High'] <= max_high) &
        (data['Low'] >= min_low) & (data['Low'] <= max_low) &
        (data['PrevClose'] >= min_prevclose) & (data['PrevClose'] <= max_prevclose)
    ]
    print(filtered_data.columns)
    # Use all four features for prediction as the model expects 4 features
    X_train_filtered = filtered_data[['Open', 'High', 'Low', 'PrevClose']]
    Y_train_filtered = filtered_data[['Close']]

    # Scale features using the same scaler as used during training
    scaler = StandardScaler()
    X_train_filtered_scaled = scaler.fit_transform(X_train_filtered)

    print(len(X_train_filtered_scaled))
    print(len(Y_train_filtered))
    y_pred_filtered = lm.predict(X_train_filtered_scaled)
    # Ensure both arrays have the same length and use the first feature for plotting
    if len(X_train_filtered_scaled) == len(Y_train_filtered) and len(X_train_filtered_scaled) > 0:
        plt.scatter(X_train_filtered['Open'], Y_train_filtered)
        plt.plot(X_train_filtered['Open'], y_pred_filtered, color='red', label='Regression Line')
        plt.title('Comparing actual and predicted prices for input data')
        plt.xlabel('PrevClose')
        plt.ylabel('Close')
        plt.savefig('/home/shoun1/airflow/dags/Stocks_ARIMA/plots/scatter_inputdata.png')
    else:
        print("X_train and Y_train have different lengths or are empty, cannot plot scatter.")
    '''rmse = mean_squared_error(y_test,y_pred)
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
    print(f"Correlation: {correlation}")'''

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
make_predictions(lm,x_train,y_train,x_test,y_test,data)
#visualize_predictions()



#sample code for processing predictions
#this was not used in the final code but can be used for further analysis
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
