from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
import mlflow
from sklearn.datasets import fetch_california_housing


def read_data():
    #data_boston = load_boston()
    #data = pd.DataFrame(data_boston.data)
    #data['PRICE'] = data_boston.target
    housing = fetch_california_housing()
    X = housing['data']
    y = housing['target']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
    return X_train, y_train, X_test, y_test


def main():
    time_now= datetime.now()
    mlflow.start_run()
    mlflow.sklearn.autolog()
    
    X_train, y_train, X_test, y_test= read_data()
    
    mlflow.log_metric('num_samples', X_train.shape[0])
    
    model= XGBRegressor()
    model.fit(X_train, y_train)
    
    prediction= model.predict(X_train)
    
    print('R^2:',metrics.r2_score(y_train, prediction))
    print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, prediction))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
    print('MAE:',metrics.mean_absolute_error(y_train, prediction))
    print('MSE:',metrics.mean_squared_error(y_train, prediction))
    print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, prediction)))
    
    
    prediction_test= model.predict(X_test)
    print('R^2:',metrics.r2_score(y_test, prediction_test))
    print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, prediction_test))*(len(y_test)-1)/(len(y_test)-X_train.shape[1]-1))
    print('MAE:',metrics.mean_absolute_error(y_test, prediction_test))
    print('MSE:',metrics.mean_squared_error(y_test, prediction_test))
    print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, prediction_test)))

    print(f'Registering  model to MLFlow')
    
    mlflow.sklearn.log_model(sk_model= model, registered_model_name= 'HousePrice', artifact_path= 'HousePrice')
    
    print(f'Saving the model')
    
    mlflow.sklearn.save_model(sk_model= model, path= f'HousePrice{str(time_now.hour)}_ {str(time_now.second)}/')
    #'./lighGBMpath' + '_'+ str(time_now.hour)+ '/'
    
    mlflow.log_metric('R2',metrics.r2_score(y_train, prediction))
    mlflow.log_metric('Adjusted R2',1 - (1-metrics.r2_score(y_train, prediction))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
    mlflow.log_metric('MAE',metrics.mean_absolute_error(y_train, prediction))
    mlflow.log_metric('MSE',metrics.mean_squared_error(y_train, prediction))
    mlflow.log_metric('RMSE',np.sqrt(metrics.mean_squared_error(y_train, prediction)))

    mlflow.end_run()
    

if __name__=='__main__':
    main()