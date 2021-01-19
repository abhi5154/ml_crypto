# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 20:11:02 2021

@author: Abhishek
"""
import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from sklearn import preprocessing
import empyrical
import pyfolio as pf
import skopt

def train_validate_test_split_random(df, train_percent=.6, validate_percent=.2, seed=None):
        
    np.random.seed(seed)
    perm         = np.random.permutation(df.index)
    m            = len(df.index)
    train_end    = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train        = df.iloc[perm[:train_end]]
    validate     = df.iloc[perm[train_end:validate_end]]
    test         = df.iloc[perm[validate_end:]]
    return train, validate, test

def train_validate_test_split_sequential(df, train_percent=.6, validate_percent=.2):
    m            = len(df.index)
    train_end    = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train        = df.iloc[0:train_end]
    validate     = df.iloc[train_end:validate_end]
    test         = df.iloc[validate_end:]
    return train, validate, test

def minmaxscaler(train_features,validate_features,test_features):
    min_max_scaler    = preprocessing.MinMaxScaler()
    x_scaled          = min_max_scaler.fit_transform(train_features)
    train_features    = pd.DataFrame(x_scaled)
    x_scaled2         = min_max_scaler.transform(validate_features)
    validate_features = pd.DataFrame(x_scaled2)
    x_scaled3         = min_max_scaler.transform(test_features)
    test_features     = pd.DataFrame(x_scaled3)

    
    return train_features,validate_features,test_features

def standardscaler(train_features,validate_features,test_features):
    standard_scaler   = preprocessing.StandardScaler()
    x_scaled          = standard_scaler.fit_transform(train_features)
    train_features    = pd.DataFrame(x_scaled)
    x_scaled2         = standard_scaler.transform(validate_features)
    validate_features = pd.DataFrame(x_scaled2)
    x_scaled3         = standard_scaler.transform(test_features)
    test_features     = pd.DataFrame(x_scaled3)

    
    return train_features,validate_features,test_features

def performance(train_features,validate_features,train_future_ret,valid_future_ret,ml_model):
    
    len_train = len(train_features)
    len_valid = len(validate_features)
    
    y_pred_train = ml_model.predict(train_features)
    y_pred_valid = ml_model.predict(validate_features)
    
    # computing returns for our fund
    ret_train  = np.where(y_pred_train == 1, train_future_ret, -train_future_ret)
    ret_valid  = np.where(y_pred_valid == 1, valid_future_ret, -valid_future_ret)
    
    total_ret = np.concatenate([ret_train,ret_valid])
    
    print("total Return ",sum(total_ret))
    #total_ret[:10]
    
    sharpe_train = empyrical.sharpe_ratio(ret_train)
    sharpe_valid = empyrical.sharpe_ratio(ret_valid)
    sharpe_all   = empyrical.sharpe_ratio(total_ret)
    
    print("sharpe_train",sharpe_train )
    print("sharpe_valid",sharpe_valid )
    print("sharpe_all",sharpe_all )
    
        
