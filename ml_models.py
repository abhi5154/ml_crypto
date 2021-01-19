# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 20:21:30 2021

@author: Abhishek
"""
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn import preprocessing
from sklearn.model_selection import KFold 

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import constants as constx
from scipy.ndimage.interpolation import shift


def Adaboostclass(train_features,train_labels,validate_features,validate_labels,train_future_ret,valid_future_ret,features):
    
    base_model = DecisionTreeClassifier(max_depth=2 ,class_weight="balanced")#min_samples_split = 0.10,min_samples_leaf = 0.10)
    clf1       = AdaBoostClassifier(base_model,n_estimators= 50 ,learning_rate= 0.50)
    clf        = clf1.fit(train_features, train_labels)
    importance = clf.feature_importances_
    print("\n", " AdaBoost Classifier \n") 

    predictions = clf.predict(train_features)
    conf_mat = confusion_matrix(train_labels, predictions)
    print(conf_mat)
    print(classification_report(train_labels, predictions))
    
    predictions2 = clf.predict(validate_features)
    conf_mat = confusion_matrix(validate_labels, predictions2)
    print(conf_mat)
    print(classification_report(validate_labels,predictions2))
    
    len_train     = len(train_features)
    y_pred_train  = clf.predict(train_features)
    tc_pred_train = np.where(y_pred_train != shift(y_pred_train, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_train     = np.where(y_pred_train == 1, train_future_ret - tc_pred_train,np.where(y_pred_train == -1,-train_future_ret - tc_pred_train,0))
    
    
    print("total Return ",np.mean(ret_train)*100*365*48,"%")
    print("total tc ",sum(tc_pred_train)*100,"%")
    sharpe_train = ret_train.mean()/ret_train.std()*np.sqrt(365*48)
    print("sharpe_train",sharpe_train )
    
    len_valid    = len(validate_features)
    y_pred_valid = clf.predict(validate_features)
    tc_pred_valid = np.where(y_pred_valid != shift(y_pred_valid, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_valid    = np.where(y_pred_valid == 1, valid_future_ret - tc_pred_valid,np.where(y_pred_valid == -1, -valid_future_ret - tc_pred_valid,0))
    
    print("total Return ",np.mean(ret_valid)*100*365*48,"%")
    print("total tc ",sum(tc_pred_valid)*100,"%")
    sharpe_valid = ret_valid.mean()/ret_valid.std()*np.sqrt(365*48)
    print("sharpe_valid",sharpe_valid)
    
    
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(25,25)) 
    plt.title('Adaboost Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    
    return clf


def RandomForestClass(train_features,train_labels,validate_features,validate_labels,train_future_ret,valid_future_ret,features):
    
    rfc1 = RandomForestClassifier(n_estimators = 50,max_depth = 2,max_features= 93,class_weight="balanced_subsample",random_state=11)#,min_samples_split = 0.05,min_samples_leaf=0.05)
    rfc  = rfc1.fit(train_features,train_labels)    
    
    importance = rfc.feature_importances_
    print("\n", " RandomForest Classifier \n ") 
    
    predictions  = rfc.predict(train_features)   
    conf_mat = confusion_matrix(train_labels, predictions)
    print(conf_mat)
    print(classification_report(train_labels, predictions))
    
    predictions2 = rfc.predict(validate_features)   
    conf_mat = confusion_matrix(validate_labels, predictions2)
    print(conf_mat)
    print(classification_report(validate_labels,predictions2))
    
    len_train     = len(train_features)
    y_pred_train  = rfc.predict(train_features)
    tc_pred_train = np.where(y_pred_train != shift(y_pred_train, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_train     = np.where(y_pred_train == 1, train_future_ret - tc_pred_train,np.where(y_pred_train == -1,-train_future_ret - tc_pred_train,0))
    
    
    print("total Return ",np.mean(ret_train)*100*365*48,"%")
    print("total tc ",sum(tc_pred_train)*100,"%")
    sharpe_train = ret_train.mean()/ret_train.std()*np.sqrt(365*48)
    print("sharpe_train",sharpe_train )
    
    len_valid    = len(validate_features)
    y_pred_valid = rfc.predict(validate_features)
    tc_pred_valid = np.where(y_pred_valid != shift(y_pred_valid, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_valid    = np.where(y_pred_valid == 1, valid_future_ret - tc_pred_valid,np.where(y_pred_valid == -1, -valid_future_ret - tc_pred_valid,0))
    
    print("total Return ",np.mean(ret_valid)*100*365*48,"%")
    print("total tc ",sum(tc_pred_valid)*100,"%")
    sharpe_valid = ret_valid.mean()/ret_valid.std()*np.sqrt(365*48)
    print("sharpe_valid",sharpe_valid)
    
    
    
    importances = rfc.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(25,25)) 
    plt.title('RandomForest Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    
    return rfc


def NeuralNetworkClass(train_features,train_labels,validate_features,validate_labels,train_future_ret,valid_future_ret,features):
    
    mlp1 = MLPClassifier(hidden_layer_sizes=(10,10),max_iter=1000,activation = 'relu')
    mlp  = mlp1.fit(train_features, train_labels)
    
    print("\n", " NeuralNetwork Classifier")
        
    predictions = mlp.predict(train_features)
    conf_mat = confusion_matrix(train_labels, predictions)
    print(conf_mat)
    print(classification_report(train_labels, predictions))
    
    predictions2 = mlp.predict(validate_features)
    conf_mat = confusion_matrix(validate_labels, predictions2)
    print(conf_mat)
    print(classification_report(validate_labels,predictions2))
    
    len_train     = len(train_features)
    y_pred_train  = mlp.predict(train_features)
    tc_pred_train = np.where(y_pred_train != shift(y_pred_train, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_train     = np.where(y_pred_train == 1, train_future_ret - tc_pred_train,np.where(y_pred_train == -1,-train_future_ret - tc_pred_train,0))
    
    
    print("total Return ",np.mean(ret_train)*100*365*48,"%")
    print("total tc ",sum(tc_pred_train)*100,"%")
    sharpe_train = ret_train.mean()/ret_train.std()*np.sqrt(365*48)
    print("sharpe_train",sharpe_train )
    
    len_valid    = len(validate_features)
    y_pred_valid = mlp.predict(validate_features)
    tc_pred_valid = np.where(y_pred_valid != shift(y_pred_valid, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_valid    = np.where(y_pred_valid == 1, valid_future_ret - tc_pred_valid,np.where(y_pred_valid == -1, -valid_future_ret - tc_pred_valid,0))
    
    print("total Return ",np.mean(ret_valid)*100*365*48,"%")
    print("total tc ",sum(tc_pred_valid)*100,"%")
    sharpe_valid = ret_valid.mean()/ret_valid.std()*np.sqrt(365*48)
    print("sharpe_valid",sharpe_valid)
    
        
    return mlp


def XGBoostClass(train_features,train_labels,validate_features,validate_labels,train_future_ret,valid_future_ret,features):

    xgbc1 = XGBClassifier(n_estimators = 50,max_depth = 2,learning_rate = 0.3)
    xgbc  = xgbc1.fit(train_features, train_labels)
    
    importance = xgbc.feature_importances_
    print("\n", " XGBoost Classifier") 
    
    predictions = xgbc.predict(train_features)
    conf_mat = confusion_matrix(train_labels, predictions)
    print(conf_mat)
    print(classification_report(train_labels, predictions))
    
    predictions2 = xgbc.predict(validate_features)
    conf_mat = confusion_matrix(validate_labels, predictions2)
    print(conf_mat)
    print(classification_report(validate_labels,predictions2))
    
    len_train     = len(train_features)
    y_pred_train  = xgbc.predict(train_features)
    tc_pred_train = np.where(y_pred_train != shift(y_pred_train, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_train     = np.where(y_pred_train == 1, train_future_ret - tc_pred_train,np.where(y_pred_train == -1,-train_future_ret - tc_pred_train,0))
    
    
    print("total Return ",np.mean(ret_train)*100*365*48,"%")
    print("total tc ",sum(tc_pred_train)*100,"%")
    sharpe_train = ret_train.mean()/ret_train.std()*np.sqrt(365*48)
    print("sharpe_train",sharpe_train )
    
    len_valid    = len(validate_features)
    y_pred_valid = xgbc.predict(validate_features)
    tc_pred_valid = np.where(y_pred_valid != shift(y_pred_valid, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_valid    = np.where(y_pred_valid == 1, valid_future_ret - tc_pred_valid,np.where(y_pred_valid == -1, -valid_future_ret - tc_pred_valid,0))
    
    print("total Return ",np.mean(ret_valid)*100*365*48,"%")
    print("total tc ",sum(tc_pred_valid)*100,"%")
    sharpe_valid = ret_valid.mean()/ret_valid.std()*np.sqrt(365*48)
    print("sharpe_valid",sharpe_valid)
    
    
    
    importances = xgbc.feature_importances_
    indices = np.argsort(importances)[(len(importances) - 10):len(importances)]
    
    plt.figure(figsize=(10,10)) 
    plt.title('Adaboost Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    

    
    return xgbc


def test_performance(clf ,test_features,test_labels,test_future_ret,features):

    predictions = clf.predict(test_features)
    conf_mat = confusion_matrix(test_labels, predictions)
    print(conf_mat)
    print(classification_report(test_labels, predictions))
    
    len_test    = len(test_features)
    y_pred_test = clf.predict(test_features)
    tc_pred_test= np.where(y_pred_test != shift(y_pred_test, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_test    = np.where(y_pred_test == 1, test_future_ret - tc_pred_test,np.where(y_pred_test == -1, -test_future_ret - tc_pred_test,0))
    
    print("total Return ",np.mean(ret_test)*100*365*48,"%")
    print("total tc ",sum(tc_pred_test)*100,"%")
    sharpe_test = ret_test.mean()/ret_test.std()*np.sqrt(365*48)
    print("sharpe_test",sharpe_test)

    


