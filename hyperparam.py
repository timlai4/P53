# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:40:52 2019

@author: Tim
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

k9_df = pd.read_csv('/u/i529/labs/lab7/Data_sets/clean_K9.data', sep = ',', header= None)

features = k9_df.loc[:,:5407]
labels = k9_df[5408]
labels = [1.0 if status == 'active' else 0.0 for status in labels]

processed_targets = np.asarray(labels)

scaler = preprocessing.StandardScaler()

processed_features = scaler.fit_transform(features)

clf = MLPClassifier(alpha=.01, hidden_layer_sizes=(1000, 1000, 500, 500),max_iter = 5000, early_stopping=True)

parameters = { 
        'batch_size': [1000,2000,3000]
	}

CV_clf = GridSearchCV(estimator = clf, param_grid = parameters, scoring = 'roc_auc', cv = 5)
CV_clf.fit(processed_features, processed_targets)
print(CV_clf.best_params_)
test = {'CV_clf.best_params_': CV_clf.best_params_}



