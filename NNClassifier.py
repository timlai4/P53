# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:40:52 2019

@author: Tim
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

k9_df = pd.read_csv('/u/i529/labs/lab7/Data_sets/clean_K9.data', sep = ',', header= None)
features = k9_df.loc[:,:5407]
labels = k9_df[5408]
labels = [1.0 if status == 'active' else 0.0 for status in labels]
print(sum(labels))
# There is a huge imbalance in the outputs.

processed_targets = np.asarray(labels)
# Rescale the features 
scaler = preprocessing.StandardScaler()
processed_features = scaler.fit_transform(features)

# Due to the unbalanced dataset, use stratified cross-validation
cv = StratifiedKFold(n_splits=10)
clf = MLPClassifier(alpha=.01, hidden_layer_sizes=(1000, 4000, 3000, 2000), max_iter=5000, early_stopping=True)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(processed_features, processed_targets):
    probas_ = clf.fit(processed_features[train], processed_targets[train]).predict_proba(processed_features[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(processed_targets[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
    print(i)
    

plt.figure(figsize = (8,8))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc.png')
print(mean_auc)

