import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

dftrain = pd.read_csv('Data/train.csv')
dftest = pd.read_csv('Data/test.csv')

#Splitting for training and testing
xtrain = dftrain.drop(['id', 'target'], axis = 1)
ytrain = dftrain['target']
xtest = dftest.drop(['id'], axis = 1)

#One Hot Encoding all features
xtrain = pd.get_dummies(xtrain, columns = xtrain.columns, dtype = 'float64', drop_first=True)
dftest = pd.get_dummies(xtest, columns = xtest.columns, dtype = 'float64', drop_first=True)

#Implementing lightGBM
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 2,
    'learning_rate': 0.3,
    'feature_fraction': 0.2,
    'is_unbalance': True
}

train_data = lgb.Dataset(xtrain,  ytrain)
test_data = lgb.Dataset(xtest, reference = train_data)
lgb_train = lgb.train(params,train_data, valid_sets = [train_data, test_data], num_boost_round=5000,)
predicted = lgb_train.predict(xtest)

#Submission
submission1 = pd.DataFrame(predicted, columns = ['target'])
submission1['id'] = dftest['id'].astype('int32')
submission1 = submission1[['id', 'target']]
submission1.to_csv('10.OneHotEncodeAllLightGBM.csv', header = True, index=False)