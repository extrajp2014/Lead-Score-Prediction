#!/usr/bin/python

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

def linear_regression(X,y,test_size=.25):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.25)

    model = LinearRegression()
    model.fit(X_train, Y_train)

    y_test_predict = model.predict(X_test)
    MSE = mean_squared_error(Y_test, y_test_predict)
    RMSE = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
    R2 = r2_score(Y_test, y_test_predict)
    m_hat = model.coef_[0]
    b_hat = model.intercept_

    print("X_train.shape:", X_train.shape)
    print("X_test.shape:", X_test.shape) 
    print("Y_train.shape:", Y_train.shape)
    print("Y_test.shape:", Y_test.shape) 
    print("y_test_predict:", y_test_predict,":list of prediction")
    print("Slope:", m_hat)
    print("Intercept:", b_hat)
    print('MSE:',MSE)
    print('RMSE:',RMSE)
    print('R^2:',R2)
    return y_test_predict

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import DataConversionWarning

def random_forest(X,y,test_size=.25):
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)
    warnings.filterwarnings(action='ignore', category=FutureWarning)
    warnings.filterwarnings(action='ignore', category=RuntimeWarning)
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    warnings.filterwarnings(action='ignore', category=UserWarning)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.25)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    model = RandomForestRegressor(n_jobs=-1)
    model.fit(X_train, Y_train)
    y_test_predict = model.predict(X_test)
    print("X_train.shape:", X_train.shape)
    print("X_test.shape:", X_test.shape) 
    print("Y_train.shape:", Y_train.shape)
    print("Y_test.shape:", Y_test.shape)
    print("y_test_predict:", y_test_predict,":list of prediction")
    print("Accuracy:",model.score(X_test, Y_test)*100, "%")
    return y_test_predict