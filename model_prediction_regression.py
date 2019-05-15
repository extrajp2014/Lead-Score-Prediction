#!/usr/bin/python

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

# Scaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

# machine learning Classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

# Models for stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, VotingClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier)

from xgboost import XGBRegressor, plot_importance

def standard_gradient_boost(X,y,test_size=.25):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # WHen you apply scaler, you turn it into float.  This turns into multi-variable classification problem into regression problem (Using classifier will cause error).  Binary-variable classification will still be classification.
    model = XGBRegressor(n_jobs=-1)
    model.fit(X_train, Y_train)
    y_test_predict = model.predict(X_test)

    name = "Standard Scaler + XGBoostRegressor"
    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    Y_train_shape = Y_train.shape
    Y_test_shape = Y_test.shape
    initial_accuracy = model.score(X_test, Y_test)*100
    average_MAE = cross_val_score(model, X_train, Y_train, cv=10, scoring='r2', verbose=1).mean()
    y_test_predict = model.predict(X_test) # list of prediction
    return name, X_train_shape, X_test_shape, Y_train_shape, Y_test_shape, initial_accuracy, average_MAE

# machine learning Regression
from sklearn.ensemble import (RandomForestRegressor)
from sklearn.preprocessing import StandardScaler
def standard_random_forest(X,y,test_size=.25):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # WHen you apply scaler, you turn it into float.  This turns into multi-variable classification problem into regression problem (Using classifier will cause error).  Binary-variable classification will still be classification.
    model = RandomForestRegressor(n_jobs=-1)
    model.fit(X_train, Y_train)
    y_test_predict = model.predict(X_test)

    name = "Standard Scaler + Random Forest Tree"
    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    Y_train_shape = Y_train.shape
    Y_test_shape = Y_test.shape
    initial_accuracy = model.score(X_test, Y_test)*100
    average_MAE = cross_val_score(model, X_train, Y_train, cv=10, scoring='r2', verbose=1).mean()
    y_test_predict = model.predict(X_test) # list of prediction
    return name, X_train_shape, X_test_shape, Y_train_shape, Y_test_shape, initial_accuracy, average_MAE

from sklearn.preprocessing import RobustScaler
def robust_random_forest(X,y,test_size=.25):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size)
    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    model = RandomForestRegressor(n_jobs=-1)
    model.fit(X_train, Y_train)
    y_test_predict = model.predict(X_test)

    name = "Robust Scaler + Random Forest Tree"
    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    Y_train_shape = Y_train.shape
    Y_test_shape = Y_test.shape
    initial_accuracy = model.score(X_test, Y_test)*100
    average_MAE = cross_val_score(model, X_train, Y_train, cv=10, scoring='r2', verbose=0).mean()
    y_test_predict = model.predict(X_test) # list of prediction
    return name, X_train_shape, X_test_shape, Y_train_shape, Y_test_shape, initial_accuracy, average_MAE

from sklearn.preprocessing import MinMaxScaler
def MinMax_random_forest(X,y,test_size=.25):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    model = RandomForestRegressor(n_jobs=-1)
    model.fit(X_train, Y_train)
    y_test_predict = model.predict(X_test)

    name = "MinMaxScaler + Random Forest Tree"
    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    Y_train_shape = Y_train.shape
    Y_test_shape = Y_test.shape
    initial_accuracy = model.score(X_test, Y_test)*100
    average_MAE = cross_val_score(model, X_train, Y_train, cv=10, scoring='r2', verbose=0).mean()
    y_test_predict = model.predict(X_test) # list of prediction
    return name, X_train_shape, X_test_shape, Y_train_shape, Y_test_shape, initial_accuracy, average_MAE

from sklearn.preprocessing import minmax_scale
def minmax_random_forest(X,y,test_size=.25):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size)
    X_train = minmax_scale(X_train)
    X_test = minmax_scale(X_test)
    model = RandomForestRegressor(n_jobs=-1)
    model.fit(X_train, Y_train)
    y_test_predict = model.predict(X_test)

    name = "minmax_scale + Random Forest Tree"
    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    Y_train_shape = Y_train.shape
    Y_test_shape = Y_test.shape
    initial_accuracy = model.score(X_test, Y_test)*100
    average_MAE = cross_val_score(model, X_train, Y_train, cv=10, scoring='r2', verbose=0).mean()
    y_test_predict = model.predict(X_test) # list of prediction
    return name, X_train_shape, X_test_shape, Y_train_shape, Y_test_shape, initial_accuracy, average_MAE

from sklearn.preprocessing import MaxAbsScaler
def MaxAbs_random_forest(X,y,test_size=.25):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size)
    scaler = MaxAbsScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    model = RandomForestRegressor(n_jobs=-1)
    model.fit(X_train, Y_train)
    y_test_predict = model.predict(X_test)

    name = "MaxAbsScaler + Random Forest Tree"
    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    Y_train_shape = Y_train.shape
    Y_test_shape = Y_test.shape
    initial_accuracy = model.score(X_test, Y_test)*100
    average_MAE = cross_val_score(model, X_train, Y_train, cv=10, scoring='r2', verbose=0).mean()
    y_test_predict = model.predict(X_test) # list of prediction
    return name, X_train_shape, X_test_shape, Y_train_shape, Y_test_shape, initial_accuracy, average_MAE

from sklearn.preprocessing import Normalizer
def Normalizer_random_forest(X,y,test_size=.25):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size)
    scaler = Normalizer()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    model = RandomForestRegressor(n_jobs=-1)
    model.fit(X_train, Y_train)


    name = "Normalizer + Random Forest Tree"
    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    Y_train_shape = Y_train.shape
    Y_test_shape = Y_test.shape
    initial_accuracy = model.score(X_test, Y_test)*100
    average_MAE = cross_val_score(model, X_train, Y_train, cv=10, scoring='r2', verbose=0).mean()
    y_test_predict = model.predict(X_test) # list of prediction
    return name, X_train_shape, X_test_shape, Y_train_shape, Y_test_shape, initial_accuracy, average_MAE

from sklearn.preprocessing import QuantileTransformer
def QuantileTransformer_random_forest(X,y,test_size=.25):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size)
    scaler = QuantileTransformer(output_distribution='normal', random_state=0)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    model = RandomForestRegressor(n_jobs=-1)
    model.fit(X_train, Y_train)
    
    name = "QuantileTransformer + Random Forest Tree"
    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    Y_train_shape = Y_train.shape
    Y_test_shape = Y_test.shape
    initial_accuracy = model.score(X_test, Y_test)*100
    average_MAE = cross_val_score(model, X_train, Y_train, cv=10, scoring='r2', verbose=0).mean()
    y_test_predict = model.predict(X_test) # list of prediction
    return name, X_train_shape, X_test_shape, Y_train_shape, Y_test_shape, initial_accuracy, average_MAE

from sklearn.preprocessing import PowerTransformer
def PowerTransformer_random_forest(X,y,test_size=.25):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size)
    scaler = PowerTransformer()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    model = RandomForestRegressor(n_jobs=-1)
    model.fit(X_train, Y_train)
    
    name = "PowerTransformer + Random Forest Tree"
    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    Y_train_shape = Y_train.shape
    Y_test_shape = Y_test.shape
    initial_accuracy = model.score(X_test, Y_test)*100
    average_MAE = cross_val_score(model, X_train, Y_train, cv=10, scoring='r2', verbose=0).mean()
    y_test_predict = model.predict(X_test) # list of prediction
    return name, X_train_shape, X_test_shape, Y_train_shape, Y_test_shape, initial_accuracy, average_MAE

def runAll(X,y, test_size=.25):
    print(standard_random_forest(X,y,test_size), ":list of score prediction")
    print("-"*100)
    print(robust_random_forest(X,y,test_size), ":list of score prediction")
    print("-"*100)
    print(MinMax_random_forest(X,y,test_size), ":list of score prediction")
    print("-"*100)
    print(minmax_random_forest(X,y,test_size), ":list of score prediction")
    print("-"*100)
    print(MaxAbs_random_forest(X,y,test_size), ":list of score prediction")
    print("-"*100)
    print(Normalizer_random_forest(X,y,test_size), ":list of score prediction")
    print("-"*100)
    print(QuantileTransformer_random_forest(X,y,test_size), ":list of score prediction")
    print("-"*100)
    print(PowerTransformer_random_forest(X,y,test_size), ":list of score prediction")
    print("-"*100)

def runAll_to_df(X,y, test_size=.25):
    '''
    return df of name, X_train.shape, X_test.shape, Y_train.shape, Y_test.shape, initial_accuracy, average_MAE, y_test_predict value for different predictive model
    '''
    
    value_df = pd.DataFrame(columns=['name', 'X_train_shape', 'X_test_shape', 'Y_train_shape', 'Y_test_shape', "initial_accuracy", "average_MAE_10_Kfold"])
    # funcs= [standard_random_forest(X,y,test_size), robust_random_forest(X,y,test_size), MinMax_random_forest(X,y,test_size), minmax_random_forest(X,y,test_size), MaxAbs_random_forest(X,y,test_size), Normalizer_random_forest(X,y,test_size), QuantileTransformer_random_forest(X,y,test_size), Normalizer_random_forest(X,y,test_size), PowerTransformer_random_forest(X,y,test_size)]
    for i in [standard_random_forest(X,y,test_size), robust_random_forest(X,y,test_size), MinMax_random_forest(X,y,test_size), minmax_random_forest(X,y,test_size), MaxAbs_random_forest(X,y,test_size), Normalizer_random_forest(X,y,test_size), QuantileTransformer_random_forest(X,y,test_size), PowerTransformer_random_forest(X,y,test_size)]:
        name, X_train_shape, X_test_shape, Y_train_shape, Y_test_shape, initial_accuracy, average_MAE = i
        value_df = pd.concat([value_df, pd.DataFrame(
            {'name':[name], 'X_train_shape':[X_train_shape], 'X_test_shape':[X_test_shape], 'Y_train_shape':[Y_train_shape], "Y_test_shape":[Y_test_shape], "initial_accuracy":[initial_accuracy], "average_MAE_10_Kfold":[average_MAE]})])
    return value_df



################################################################### 
#     print(model_prediction.standard_random_forest(X,y,0.25), ":list of score prediction")
#     print("-"*100)
#     print(model_prediction.robust_random_forestobust_random_forest(X,y,test_size), ":list of score prediction")
#     print("-"*100)
#     print(model_prediction.MinMax_random_forest(X,y,test_size), ":list of score prediction")
#     print("-"*100)
#     print(model_prediction.minmax_random_forest(X,y,test_size), ":list of score prediction")
#     print("-"*100)
#     print(model_prediction.MaxAbs_random_forest(X,y,test_size), ":list of score prediction")
#     print("-"*100)
#     print(model_prediction.Normalizer_random_forest(X,y,test_size), ":list of score prediction")
#     print("-"*100)
#     print(model_prediction.QuantileTransformer_random_forest(X,y,test_size), ":list of score prediction")
#     print("-"*100)
#     print(model_prediction.PowerTransformer_random_forest(X,y,test_size), ":list of score prediction")
################################################################### 

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

# Models for stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, VotingClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier)
