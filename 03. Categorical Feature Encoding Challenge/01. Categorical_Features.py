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

#Dataset Summary
summary = pd.DataFrame(dftrain.columns)
summary['Uniques'] = dftrain.nunique().values

#Encoding Binary Features (If we used Label_Encoding, would have to do it individually for every column)
#OneHot would create multiple columns.
dftrain['bin_3'] = dftrain['bin_3'].apply(lambda x: 1 if x == 'T' else 0)
dftrain['bin_4'] = dftrain['bin_4'].apply(lambda x: 1 if x == 'Y' else 0)

dftest['bin_3'] = dftest['bin_3'].apply(lambda x: 1 if x == 'T' else 0)
dftest['bin_4'] = dftest['bin_4'].apply(lambda x: 1 if x == 'Y' else 0)


#Encoding Nominal Features(Low Cardinality)
#pd.get_dummies() takes a lot of computational power and should not be used for columns with high cardinality
dftrain = pd.get_dummies(dftrain, columns = dftrain.columns[6:11], dtype = 'int', drop_first=True)

dftest = pd.get_dummies(dftest, columns = dftest.columns[6:11], dtype = 'int', drop_first=True)

#Encoding Ordinal Features(Low Cardinality)
# labelenc = LabelEncoder()
# for ordinalcolumn in ['ord_1', 'ord_2', 'ord_3', 'ord_4']:
#     dftrain.loc[:, ordinalcolumn] = labelenc.fit_transform(dftrain.loc[:, ordinalcolumn])
#     dftest.loc[:, ordinalcolumn] = labelenc.fit_transform(dftest.loc[:, ordinalcolumn])

#Label encoding will crete problems in priority

dftrain.ord_1.replace(to_replace = ['Novice', 'Contributor','Expert', 'Master', 'Grandmaster'],
                         value = [0, 1, 2, 3, 4], inplace = True)

dftrain.ord_2.replace(to_replace = ['Freezing', 'Cold', 'Warm', 'Hot','Boiling Hot', 'Lava Hot'],
                         value = [0, 1, 2, 3, 4, 5], inplace = True)

dftrain.ord_3.replace(to_replace = ['a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'],
                         value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], inplace = True)

dftrain.ord_4.replace(to_replace = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','J', 'K', 'L', 'M', 'N', 'O',
                                     'P', 'Q', 'R','S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
                         value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                  22, 23, 24, 25], inplace = True)

dftest.ord_1.replace(to_replace = ['Novice', 'Contributor','Expert', 'Master', 'Grandmaster'],
                         value = [0, 1, 2, 3, 4], inplace = True)

dftest.ord_2.replace(to_replace = ['Freezing', 'Cold', 'Warm', 'Hot','Boiling Hot', 'Lava Hot'],
                         value = [0, 1, 2, 3, 4, 5], inplace = True)

dftest.ord_3.replace(to_replace = ['a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'],
                         value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], inplace = True)

dftest.ord_4.replace(to_replace = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','J', 'K', 'L', 'M', 'N', 'O',
                                     'P', 'Q', 'R','S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
                         value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                  22, 23, 24, 25], inplace = True)

#Encoding Date Values using Cyclic Feature Engineering.
'''Suppose our data has to distinguish sales during years. In such a case, lets say we have a column with the date and 
month (eg. Dec 31 and Jan 1 are two dates). For such dates our model will think very differently as the dates are in 
opposite months, however in reality they are very similar. So we use cyclic engineering in our model.'''

dftrain['day_sin'] = np.sin(2*np.pi*dftrain['day']/7)
dftrain['day_cos'] = np.cos(2*np.pi*dftrain['day']/7)
dftrain['month_sin'] = np.sin(2*np.pi*dftrain['month']/12)
dftrain['month_cos'] = np.cos(2*np.pi*dftrain['month']/12)
dftrain.drop(['day', 'month'], axis=1, inplace = True)

dftest['day_sin'] = np.sin(2*np.pi*dftest['day']/7)
dftest['day_cos'] = np.cos(2*np.pi*dftest['day']/7)
dftest['month_sin'] = np.sin(2*np.pi*dftest['month']/12)
dftest['month_cos'] = np.cos(2*np.pi*dftest['month']/12)
dftest.drop(['day', 'month'], axis=1, inplace = True)

#Encoding Nominal Features(High Cardinality)
#We can use 'Hashing Trick' or 'Encoding with Frequency' for this

#Encoding using Frequency of the value in the DataFrame
for nominalcolumn in ['nom_5','nom_6','nom_7','nom_8', 'nom_9']:
    frequencydf = (dftrain.groupby(nominalcolumn).size())/len(dftrain)
    dftrain[nominalcolumn] = dftrain[nominalcolumn].apply(lambda x: frequencydf[x])

    frequencydf = (dftest.groupby(nominalcolumn).size()) / len(dftest)
    dftest[nominalcolumn] = dftest[nominalcolumn].apply(lambda x: frequencydf[x])
#     print(len(dftrain[nominalcolumn].unique()))

#Encoding using Hashing.
# for nominalcolumn in ['nom_5','nom_6','nom_7','nom_8', 'nom_9']:
#     dftrain[nominalcolumn] = dftrain[nominalcolumn].apply(lambda x: hash(x)%500)
#     print(len(dftrain[nominalcolumn].unique()))

#Since the unique values are less after encodng with frequency, we use that.

#Encoding ordinal Features(High Cardinality)
#Adding indices of both characters
# dftrain['ord_5'] = dftrain['ord_5'].apply(lambda x:sum([(string.ascii_letters.find(letter)+1) for letter in x]))
# dftest['ord_5'] = dftest['ord_5'].apply(lambda x:sum([(string.ascii_letters.find(letter)+1) for letter in x]))

#Sort values by string of ord_5(Best)
ord_5 = sorted(list(set(dftrain['ord_5'].values)))
ord_5 = dict(zip(ord_5, range(len(ord_5))))
dftrain.loc[:, 'ord_5'] = dftrain['ord_5'].apply(lambda x: ord_5[x]).astype(float)

ord_5_1 = sorted(list(set(dftest['ord_5'].values)))
ord_5_1 = dict(zip(ord_5_1, range(len(ord_5_1))))
dftest.loc[:, 'ord_5'] = dftest['ord_5'].apply(lambda x: ord_5_1[x]).astype(float)

#Encoding by creating 2 columns, each corresponding to index of character
# dftrain['ord_5_oe1'] = dftrain['ord_5'].apply(lambda x:(string.ascii_letters.find(x[0])+1))
# dftrain['ord_5_oe2'] = dftrain['ord_5'].apply(lambda x:(string.ascii_letters.find(x[1])+1))
#
# dftest['ord_5_oe1'] = dftest['ord_5'].apply(lambda x:(string.ascii_letters.find(x[0])+1))
# dftest['ord_5_oe2'] = dftest['ord_5'].apply(lambda x:(string.ascii_letters.find(x[1])+1))

dftrain = dftrain.astype('float64')
dftest = dftest.astype('float64')

#Splitting for training and testing
xtrain = dftrain.drop(['id', 'target'], axis = 1)
ytrain = dftrain['target']
xtest = dftest.drop(['id'], axis = 1)


#Standardising/Normalising
# sc = StandardScaler()
# xtrain = sc.fit_transform(xtrain)
# xtest = sc.transform(xtest)

#Training model
#LogisticRegression
# lr = LogisticRegression()
# lr.fit(xtrain, ytrain)
# scores = cross_val_score(lr, xtrain, ytrain, cv = 3, scoring = 'roc_auc')
# predicted = lr.predict(xtest)
# print(scores)

#LightGBM
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
lgb_train = lgb.train(params,train_data, valid_sets = [train_data, test_data], num_boost_round=7000,)
predicted = lgb_train.predict(xtest)


#RandomForest
# rfc = RandomForestClassifier(n_estimators = 1000)
# rfc.fit(xtrain, ytrain)
# predicted = rfc.predict(xtest)

#XGBoost
# X_train, X_val, y_train, y_val = train_test_split(xtrain, ytrain, test_size = 0.01, random_state=2019)
#
# model = XGBClassifier(objective ='binary:logistic', colsample_bytree = 0, learning_rate = 0.2,
#                 max_depth = 16, n_estimators = 2500, scale_pos_weight=2, random_state=2019)
#
# model.fit(X_train,y_train)
# pred = model.predict_proba(X_val)[:,1]
# score = roc_auc_score(y_val ,pred)
# print(score)
#
# predicted = model.predict_proba(xtest)[:,1]

# Submission
submission1 = pd.DataFrame(predicted, columns = ['target'])
submission1['id'] = dftest['id'].astype('int32')
submission1 = submission1[['id', 'target']]
#
submission1.to_csv('10.LightGBM_ManualOrdinal_Submission.csv', header = True, index=False)