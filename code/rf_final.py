# -*- coding: utf-8 -*-

"""
code to make a prediction by random forest from sklearn.
it can work in test mode if you change TEST to 1.
number of iteration in result can be changed.
parameters were copied from rf_grid_search.py.
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

TEST = 0 # or make a submission file
ITER = 100 # number of iterations

def Gini(y_true, y_pred):
    n_samples = y_true.shape[0]
    
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)
    
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    return G_pred/G_true

print('Reading data...')

train = pd.read_csv('../input/train.csv', index_col=0)
test = pd.read_csv('../input/test.csv', index_col=0)

labels = train.Hazard
train.drop('Hazard', axis=1, inplace=True)
test_ind = test.index
train = np.array(train)
test = np.array(test)

for i in range(train.shape[1]):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[:,i]) + list(test[:,i]))
    train[:,i] = lbl.transform(train[:,i])
    test[:,i] = lbl.transform(test[:,i])

train = train.astype(float)
test = test.astype(float)

rgr = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='log2', max_leaf_nodes=None, min_samples_leaf=7,
           min_samples_split=8, min_weight_fraction_leaf=0.0,
           n_estimators=1000, n_jobs=-1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)

if (TEST):
    Xtr, Xts, ytr, yts = train_test_split(train, labels, test_size=.2)
    
    print('Train RF for test...')
    rgr.fit(Xtr, ytr)
    
    print('Predicting for test...')
    ysp = rgr.predict(Xts)
    
    print('Validation Score: ', Gini(yts, ysp))

else:
    yp = np.zeros(test.shape[0])
    
    print('Train and predict...')
    for i in range(ITER):
        rgr.fit(train,labels)
        yp += rgr.predict(test) / ITER
    
    preds = pd.DataFrame({"Id": test_ind, "Hazard": yp})
    preds = preds.set_index('Id')
    preds.to_csv('../output/rf.csv')