
# import warnings
# warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import feature_label_maker as flm
import train_test_splits as tts
import ml_models as mlm
import random

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn import preprocessing
from sklearn.model_selection import KFold 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

base_folder = "C://Users//Abhishek//Desktop//WQU//CAPSTONE//PROJECT"
data_folder = "C://Users//Abhishek//Desktop//WQU//CAPSTONE//PROJECT//DATA"

time_frame   = "30"
asset_name   = "BTCUSDT"
asset_type   = "SPOT"
data_folder2 = data_folder + "//" + asset_type + "//" + time_frame 

os.chdir(data_folder2)
total_data               = pd.read_csv(asset_name+ ".csv")
total_data['close_time'] = pd.to_datetime(total_data['close_time'])
total_data['open_time']  = pd.to_datetime(total_data['open_time'])

total_data['close_time'] = total_data['close_time'].dt.round('30min')
total_data['open_time']  = total_data['open_time'].dt.round('30min')
total_data               = total_data.set_index('close_time' )
os.chdir(base_folder)

total_data = total_data[['open','high','low','close','volume','trades']]

labels1       = total_data.loc[:,'close'].shift(-1)/total_data.loc[:,'close'].shift(0) -1
labels2       = np.where(labels1>0,1,np.where(labels1<0,-1,np.NAN))
frame         = pd.DataFrame({'label' : labels2,'future_ret':labels1})
frame         = frame.set_index(total_data.index)
frame.columns = ['label','future_ret']


total_data_return   = total_data.shift(0)/total_data.shift(1) - 1
technical_features  = flm.technical_indicators(total_data)
derived_features    = flm.price_derivations(total_data)
previous_features   = flm.previous_returns(total_data)
season_features     = flm.seasonal_features(total_data)

dfx = total_data_return

dfx = dfx.merge(technical_features,left_on='close_time', right_on='close_time')
dfx = dfx.merge(derived_features,left_on='close_time', right_on='close_time')
dfx = dfx.merge(previous_features,left_on='close_time', right_on='close_time')
dfx = dfx.merge(season_features,left_on='close_time', right_on='close_time')
dfx = dfx.merge(frame,left_on='close_time', right_on='close_time')

dfx = dfx.replace([np.inf, -np.inf], np.nan)
dfx = dfx.dropna()
dfx.reset_index(drop=True, inplace=True)


train ,validate ,test = tts.train_validate_test_split_sequential(dfx ,0.6 ,0.15)
#train ,validate ,test = tts.train_validate_test_split_random(dfx ,0.6 ,0.15)

features = dfx.columns.tolist()
features = [e for e in features if e not in ('label', 'future_ret')]

train_features    = train[features]
validate_features = validate[features]
test_features     = test[features]

train_labels     = train['future_ret']
validate_labels  = validate['future_ret']
test_labels      = test['future_ret']

train_future_ret = train['future_ret']
valid_future_ret = validate['future_ret']
test_future_ret  = test['future_ret']

train_feature_list = list(train_features.columns)

print(features)

train_features , validate_features = tts.minmaxscaler(train_features,validate_features)

# base_model = DecisionTreeRegressor(criterion='mse',splitter = 'best',max_depth=2,min_samples_split= 0.10,min_samples_leaf= 0.10)
# clf1       = AdaBoostRegressor(base_model,n_estimators= 20)

clf1 = RandomForestRegressor(n_estimators= 50,max_depth = 2,min_samples_split= 0.01,min_samples_leaf=0.01 ,max_features= 75)


clf        = clf1.fit(train_features, train_labels)
importance = clf.feature_importances_
#print(importance) 

predictions = clf.predict(train_features)
predictions2 = clf.predict(validate_features)
print("r2 train ",r2_score(train_labels, predictions))
print("r2 valid ",r2_score(validate_labels,predictions2))

thresh    = 0.0005

len_train    = len(train_features)
y_pred_train = clf.predict(train_features)
ret_train  = np.where(y_pred_train >thresh, train_future_ret,np.where(y_pred_train < -thresh, -train_future_ret,0))

print("total Return ",np.mean(ret_train)*100*365*48,"%")
sharpe_train = ret_train.mean()/ret_train.std()*np.sqrt(365*48)
print("sharpe_train",sharpe_train )

len_valid    = len(validate_features)
y_pred_valid = clf.predict(validate_features)
ret_valid    = np.where(y_pred_valid >thresh, valid_future_ret,np.where(y_pred_valid < -thresh, -valid_future_ret,0))

print("total Return ",np.mean(ret_valid)*100*365*48,"%")
sharpe_valid = ret_valid.mean()/ret_valid.std()*np.sqrt(365*48)
print("sharpe_valid",sharpe_valid)


importances = clf.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(30,10)) 
plt.title('Adaboost Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
