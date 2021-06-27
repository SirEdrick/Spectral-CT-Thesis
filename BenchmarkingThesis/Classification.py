# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 10:32:21 2021

@author: mbust, bruun
"""

import pandas as pd
import numpy as np

#--------------------setup the data-----------------------
#-labels
#-df type file

datapath_head = "G:/StudieBackup/Test_dataset/Sample_06062018_Fluids"
#datapath_head = "G:/StudieBackup/Test_dataset/Sample_06062018_Fruits"
#datapath_head = "G:/StudieBackup/Test_dataset/Sample_06062018_NonThreat"
#datapath_head = "G:/StudieBackup/Test_dataset/Sample_06062018_Threat"
#datapath_head = "G:/StudieBackup/Test_dataset/Sample_23012018"
#datapath_head = "G:/StudieBackup/Test_dataset/Sample_24012018"

labels = ["acetone","h2o","h2o2","nitric_acid","olive_oil","whiskey"]
#labels = ["grapes","apple","tropicalJuice","citrusJuice"]
#labels = ["Nivea_sun_lotion_pf50+", "Olive_oil", "Toothpaste","h2o","Cien_hand_creme"]
#labels = ["acetone","C4","h2o2","methanol"]
#labels = ["h2o", "h2o2", "whiskey","hand_cream","toothaste","C4","aluminium_rod"]
#labels = ["aluminium_rod", "C4", "hand_cream","h2o2","toothaste","h2o","whiskey"]

#-Labels Load for each type

# ART9 ------------

file1 = datapath_head + "/processed/segmentedART9/"+'labels_all.txt'

labels_all1 = pd.read_csv(file1, delimiter = "\t", header=None)
flat_labels1 = labels_all1.to_numpy().flatten()
label_names_set1 = set(flat_labels1)

label_ids1 = [labels.index(x) for x in flat_labels1]
label_ids_set1 = set(label_ids1)

# ART74 -----------

file2 = datapath_head + "/processed/segmentedART74/"+'labels_all.txt'

labels_all2 = pd.read_csv(file2, delimiter = "\t", header=None)
flat_labels2 = labels_all2.to_numpy().flatten()
label_names_set2 = set(flat_labels2)

label_ids2 = [labels.index(x) for x in flat_labels2]

# NN9 ------------

file3 = datapath_head + "/processed/segmentedNN/"+'labels_all.txt'

labels_all3 = pd.read_csv(file3, delimiter = "\t", header=None)
flat_labels3 = labels_all3.to_numpy().flatten()
label_names_set3 = set(flat_labels3)

label_ids3 = [labels.index(x) for x in flat_labels3]
label_ids_set3 = set(label_ids3)

#-LAC Load for each type

# ART9 ------------

print("Loading LAC data ART9...")
data_file1 = datapath_head + "/processed/segmentedART9/"+'LAC_all.csv'
        
columns1 = list(np.arange(1,33))

data_allART9 = pd.read_csv(data_file1, header=None)
data_allART9.columns = columns1
data_allART9['ids'] = label_ids1

print("LAC data loaded. Number of voxels: {}".format(len(data_allART9)))

print("ART9 created here's the distribution of each label:")
for i in range(len(labels)):
    print(str(i) + " ({:}): ".format(labels[i]) + str(len(data_allART9.ids[data_allART9.ids == i])) + " voxels")

# ART74 -----------

print("Loading LAC data ART74...")
data_file2 = datapath_head + "/processed/segmentedART74/" + 'LAC_all.csv'

columns2 = list(np.arange(1, 33))

data_allART74 = pd.read_csv(data_file2, header=None)
data_allART74.columns = columns2
data_allART74['ids'] = label_ids2

print("LAC data loaded. Number of voxels: {}".format(len(data_allART74)))

# The distribution is the same for all LACs
#print("ART74 created, here's the distribution of each label:")
#for i in range(len(labels)):
#    print(str(i) + " ({:}): ".format(labels[i]) + str(len(data_allART74.ids[data_allART74.ids == i])) + " voxels")

# NN9 ------------

print("Loading LAC data NN9...")
data_file3 = datapath_head + "/processed/segmentedNN/" + 'LAC_all.csv'
columns3 = list(np.arange(1, 33))

data_allNN = pd.read_csv(data_file3, header=None)
data_allNN.columns = columns3
data_allNN['ids'] = label_ids3

print("LAC data loaded. Number of voxels: {}".format(len(data_allNN)))

# The distribution is the same for all LACs
#print("NN9 created, here's the distribution of each label:")
#for i in range(len(labels)):
#    print(str(i) + " ({:}): ".format(labels[i]) + str(len(data_allNN.ids[data_allNN.ids == i])) + " voxels")

print("Creating sklearning model...")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score

# Splits data into three data sets for ART9, ART74 and NN9
X1, y1 = data_allART9.iloc[:,:-1],label_ids1
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=0)
X2, y2 = data_allART74.iloc[:,:-1],label_ids2
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=0)
X3, y3 = data_allNN.iloc[:,:-1],label_ids3
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.2, random_state=0)

# Creates and Fits a RandomForestClassifier for each data set
rfc1 = RandomForestClassifier()
rfc2 = RandomForestClassifier()
rfc3 = RandomForestClassifier()
model1 = rfc1.fit(X_train1, y_train1)
model2 = rfc2.fit(X_train2, y_train2)
model3 = rfc3.fit(X_train3, y_train3)

print()
print("Testing Model ART 9 for test set")

y1_pred1 = model1.predict(X_test1)
y1_recall = recall_score(y_test1, y1_pred1, average=None)
y1_roc = roc_auc_score(y_test1, model1.predict_proba(X_test1), multi_class='ovo')
average_precision1 = precision_score(y_test1, y1_pred1, average=None)
print("Number of mislabeled points out of a total %d points : %d" % (X_test1.shape[0], (y_test1 != y1_pred1).sum()))
score1 = model1.score(X_test1, y_test1)
print("Accuracy: ")
print(score1)
print('Precision score: ')
print(average_precision1)
print("Recall: ")
print(y1_recall)
f1_1 = f1_score(y_test1, y1_pred1, average=None)
print("F1: ")
print(f1_1)
print("ROC: ")
print(y1_roc)


print()
print("Testing Model ART 74 for test set")

y2_pred2 = model2.predict(X_test2)
y2_recall = recall_score(y_test2, y2_pred2, average=None)
y2_roc = roc_auc_score(y_test2, model2.predict_proba(X_test2), multi_class='ovo')
average_precision2 = precision_score(y_test2, y2_pred2, average=None)
print("Number of mislabeled points out of a total %d points : %d" % (X_test2.shape[0], (y_test2 != y2_pred2).sum()))
score22 = model2.score(X_test2,y_test2)
print("Accuracy: ")
print(score22)
print('Precision score: ')
print(average_precision2)
print("Recall: ")
print(y2_recall)
f1_2 = f1_score(y_test2, y2_pred2, average=None)
print("F1: ")
print(f1_2)
print("ROC: ")
print(y2_roc)

print()
print("Testing Model NN 9 for test set")

y3_pred3 = model3.predict(X_test3)
y3_recall = recall_score(y_test3, y3_pred3, average=None)
y3_roc = roc_auc_score(y_test3, model3.predict_proba(X_test3), multi_class='ovo')
average_precision3 = precision_score(y_test3, y3_pred3, average=None)
print("Number of mislabeled points out of a total %d points : %d" % (X_test3.shape[0], (y_test3 != y3_pred3).sum()))
score33 = model3.score(X_test3,y_test3)
print("Accuracy: ")
print(score33)
print('Precision score: ')
print(average_precision3)
print("Recall: " )
print(y3_recall)
f1_3 = f1_score(y_test3, y3_pred3, average=None)
print("F1: ")
print(f1_3)
print("ROC: " )
print(y3_roc)


import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

# Plots the Confussion Matrices

plot_confusion_matrix(rfc1, X_test1, y_test1, display_labels=labels, normalize='true')
plt.savefig("confusion_RFC_ART9_normalized.pdf")
plot_confusion_matrix(rfc1, X_test1, y_test1, display_labels=labels)
plt.savefig("confusion_RFC_ART9.pdf")

plot_confusion_matrix(rfc2, X_test2, y_test2, display_labels=labels, normalize='true')
plt.savefig("confusion_RFC_ART74_normalized.pdf")
plot_confusion_matrix(rfc2, X_test2, y_test2, display_labels=labels)
plt.savefig("confusion_RFC_ART74.pdf")

plot_confusion_matrix(rfc3, X_test3, y_test3, display_labels=labels, normalize='true')
plt.savefig("confusion_RFC_NN9_normalized.pdf")
plot_confusion_matrix(rfc3, X_test3, y_test3, display_labels=labels)
plt.savefig("confusion_RFC_NN9.pdf")

plt.show()