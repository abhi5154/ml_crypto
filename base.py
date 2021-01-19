# import warnings
# warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import random
import constants as constx
import feature_label_maker as flm
import data_loading as data_loading_module
import train_test_splits as tts
import ml_models as mlm


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from scipy.ndimage.interpolation import shift

from sklearn import preprocessing
from sklearn.model_selection import KFold 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

np.random.seed(constx.seedx)


total_data    = data_loading_module.base_data_loader((constx.asset_name), constx.asset_type, constx.time_frame)
#label_frames = flm.label_binary(total_data) 
label_data    = flm.label_multiple(total_data ,constx.thresh) 

dfx = flm.load_all_feature_types(total_data)
dfx = dfx.merge(label_data,left_on='close_time', right_on='close_time')

dfx = dfx.replace([np.inf, -np.inf], np.nan)
dfx = dfx.dropna()
dfx.reset_index(drop=True, inplace=True)

train ,validate ,test = tts.train_validate_test_split_sequential(dfx ,constx.train_percent ,constx.valid_percent)
#train ,validate ,test = tts.train_validate_test_split_random(dfx ,constx.train_percent ,constx.valid_percent ,seed = seedx)

features = dfx.columns.tolist()
features = [e for e in features if e not in ('label', 'future_ret')]

train_features    = train[features]
validate_features = validate[features]
test_features     = test[features]

train_labels     = train['label']
validate_labels  = validate['label']
test_labels      = test['label']

train_future_ret = train['future_ret']
valid_future_ret = validate['future_ret']
test_future_ret  = test['future_ret']

train_feature_list = list(train_features.columns)

print("\n",features)

#props         = 1/train_labels.value_counts()/sum(1/train_labels.value_counts())
#sample_weights = np.where(train_labels == train_labels.value_counts()[0] ,props[0],np.where(train_labels == train_labels.value_counts()[1] ,props[1],props[-1]))

#train_features , validate_features = tts.minmaxscaler(train_features,validate_features)
train_features , validate_features ,test_features = tts.standardscaler(train_features, validate_features ,test_features)


clf  = mlm.Adaboostclass(train_features, train_labels, validate_features, validate_labels, train_future_ret, valid_future_ret, features)
#clf  = mlm.RandomForestClass(train_features, train_labels, validate_features, validate_labels, train_future_ret, valid_future_ret, features)
#clf  = mlm.XGBoostClass(train_features, train_labels, validate_features, validate_labels, train_future_ret, valid_future_ret, features)
#clf  = mlm.NeuralNetworkClass(train_features, train_labels, validate_features, validate_labels, train_future_ret, valid_future_ret, features)

print ("\n", "        TEST PERFORMANCE  ")
test_performance = mlm.test_performance(clf ,test_features,test_labels,test_future_ret,features)

importances = clf.feature_importances_
indices = np.argsort(importances)[(len(importances) - 10):len(importances)]

plt.figure(figsize=(10,8)) 
plt.title('AdaBoost Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


