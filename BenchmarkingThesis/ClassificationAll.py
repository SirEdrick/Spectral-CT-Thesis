# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 14:07:54 2021

@author: mbust, bruun
"""

import pandas as pd
import numpy as np

#--------------------setup the data-----------------------
#-labels
#-df type file
datapath_head1 = "G:/StudieBackup/Test_dataset/Sample_06062018_Fluids"
file1 = datapath_head1 + "/processed/segmented/"+'labels_all.txt'
datapath_head2 = "G:/StudieBackup/Test_dataset/Sample_06062018_Fruits"
file2 = datapath_head2 + "/processed/segmented/"+'labels_all.txt'
datapath_head3 = "G:/StudieBackup/Test_dataset/Sample_06062018_NonThreat"
file3 = datapath_head3 + "/processed/segmented/"+'labels_all.txt'
datapath_head4 = "G:/StudieBackup/Test_dataset/Sample_06062018_Threat"
file4 = datapath_head4 + "/processed/segmented/"+'labels_all.txt'
datapath_head5 = "G:/StudieBackup/Test_dataset/Sample_23012018"
file5 = datapath_head5 + "/processed/segmented/"+'labels_all.txt'
datapath_head6 = "G:/StudieBackup/Test_dataset/Sample_24012018"
file6 = datapath_head6 + "/processed/segmented/"+'labels_all.txt'

all_datapath_heads = [datapath_head1,datapath_head2,datapath_head3,datapath_head4,datapath_head5,datapath_head6]
all_files = [file1,file2,file3,file4,file5,file6]
all_filename=["Sample_06062018_Fluids","Sample_06062018_Fruits","Sample_06062018_NonThreat","Sample_06062018_Threat",
              "Sample_23012018","Sample_24012018"]

all_label_names = [[],[],[],[],[],[]]
all_label_names_set = [[],[],[],[],[],[]]
all_label_ids = [[],[],[],[],[],[]]
all_label_ids_set = [[],[],[],[],[],[]]

labels1 = ["acetone","h2o","h2o2","nitric_acid","olive_oil","whiskey"]
labels2 = ["grapes","apple","tropicalJuice","citrusJuice"]
labels3 = ["Nivea_sun_lotion_pf50+", "Olive_oil", "Toothpaste","h2o","Cien_hand_creme"]
labels4 = ["acetone","C4","h2o2","methanol"]
labels5 = ["h2o", "h2o2", "whiskey","hand_cream","toothaste","C4","aluminium_rod"]
labels6 = ["aluminium_rod", "C4", "hand_cream","h2o2","toothaste","h2o","whiskey"]
labels = labels1+labels2+labels3+labels4+labels5+labels6
labels = list(dict.fromkeys(labels))

#-------------Setup labels & ids ---------------------
for v in range(len(all_files)):
    file = all_files[v]
    filename = all_filename[v]
    print("Seting-up file: " + filename)
    
    
    labels_all = pd.read_csv(file, delimiter = "\t", header=None)
    flat_labels = labels_all.to_numpy().flatten()
    label_names_set = set(flat_labels)
    label_ids = [labels.index(x) for x in flat_labels]
    label_ids_set = set(label_ids)
    
    print(label_names_set)
    print(label_ids_set)
    print()
    
    all_label_names[v] = flat_labels 
    all_label_names_set[v] = label_names_set
    all_label_ids[v] = label_ids
    all_label_ids_set[v] = label_ids_set

#------------Load LAVs data-------------------------------

# ART9 ----------------------

all_data_all_ART9 = [[],[],[],[],[],[]]

for v in range(len(all_files)):
    file = all_files[v]
    filename = all_filename[v]
    print("Loading data for file: "+filename)

    data_file_ART9 = all_datapath_heads[v] + "/processed/segmentedART9/"+'LAC_all.csv'
        
    columns = list(np.arange(1,33))

    data_all_ART9 = pd.read_csv(data_file_ART9, header=None)
    data_all_ART9.columns = columns
    data_all_ART9['ids'] = all_label_ids[v]
    
    all_data_all_ART9[v] = data_all_ART9
    
data_master_ART9 = pd.concat([all_data_all_ART9[0], all_data_all_ART9[1], all_data_all_ART9[2], all_data_all_ART9[3], all_data_all_ART9[4],
                         all_data_all_ART9[5]], axis=0)

print("Data master ART9 created (combination of all dataframes), here's the distribution of each label:")
for i in range(19):
    print(str(i)+" ({:}): ".format(labels[i])+str(len(data_master_ART9.ids[data_master_ART9.ids==i]))+" voxels")

# ART74--------------

all_data_all_ART74 = [[], [], [], [], [], []]

for v in range(len(all_files)):
    file = all_files[v]
    filename = all_filename[v]
    print("Loading data for file: " + filename)

    data_file_ART74 = all_datapath_heads[v] + "/processed/segmentedART74/" + 'LAC_all.csv'

    columns = list(np.arange(1, 33))

    data_all_ART74 = pd.read_csv(data_file_ART74, header=None)
    data_all_ART74.columns = columns
    data_all_ART74['ids'] = all_label_ids[v]

    all_data_all_ART74[v] = data_all_ART74

data_master_ART74 = pd.concat(
    [all_data_all_ART74[0], all_data_all_ART74[1], all_data_all_ART74[2], all_data_all_ART74[3], all_data_all_ART74[4],
     all_data_all_ART74[5]], axis=0)

print("Data master ART74 created (combination of all dataframes), here's the distribution of each label:")
for i in range(19):
    print(str(i) + " ({:}): ".format(labels[i]) +
          str(len(data_master_ART74.ids[data_master_ART74.ids == i])) + " voxels")

# NN9 ---------------

all_data_all_NN = [[], [], [], [], [], []]

for v in range(len(all_files)):
    file = all_files[v]
    filename = all_filename[v]
    print("Loading data for file: " + filename)

    data_file = all_datapath_heads[v] + "/processed/segmentedNN/" + 'LAC_all.csv'

    columns = list(np.arange(1, 33))

    data_all_NN = pd.read_csv(data_file, header=None)
    data_all_NN.columns = columns
    data_all_NN['ids'] = all_label_ids[v]

    all_data_all_NN[v] = data_all_NN

data_master_NN = pd.concat(
    [all_data_all_NN[0], all_data_all_NN[1], all_data_all_NN[2], all_data_all_NN[3], all_data_all_NN[4],
     all_data_all_NN[5]], axis=0)

print("Data master NN9 created (combination of all dataframes), here's the distribution of each label:")
for i in range(19):
    print(str(i) + " ({:}): ".format(labels[i]) + str(len(data_master_NN.ids[data_master_NN.ids == i])) + " voxels")

#-------------setup the data - done -----------------------

print("Creating sklearning model...")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score

# Splits data into three data sets for ART9, ART74 and NN9
y_all = all_label_ids[0]+all_label_ids[1]+all_label_ids[2]+all_label_ids[3]+all_label_ids[4]+all_label_ids[5]
print("Spliting ART9...")
X1, y1 = data_master_ART9.iloc[:,:-1],y_all
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=0)
print("Spliting ART74...")
X2, y2 = data_master_ART74.iloc[:,:-1],y_all
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=0)
print("Spliting NN9...")
X3, y3 = data_master_ART9.iloc[:,:-1],y_all
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.2, random_state=0)

# Creates and Fits a RandomForestClassifier for each data set
print("Fitting ART9...")
rfc1 = RandomForestClassifier()
model1 = rfc1.fit(X_train1, y_train1)
print("Fitting ART74..")
rfc2 = RandomForestClassifier()
model2 = rfc2.fit(X_train2, y_train2)
print("Fitting NN9...")
rfc3 = RandomForestClassifier()
model3 = rfc3.fit(X_train3, y_train3)

print("ART9 all results:")
print()
y_pred1 = model1.predict(X_test1)
y_recall1 = recall_score(y_test1, y_pred1, average=None)
y_roc1 = roc_auc_score(y_test1, model1.predict_proba(X_test1), multi_class='ovo')
average_precision1 = precision_score(y_test1, y_pred1, average=None)
print("Number of mislabeled points out of a total %d points : %d" % (X_test1.shape[0], (y_test1 != y_pred1).sum()))
score1 = model1.score(X_test1, y_test1)
print("Accuracy: ")
print(score1)
print('Precision score: ')
print(average_precision1)
print("Recall: ")
print(y_recall1)
f1_1 = f1_score(y_test1, y_pred1, average=None)
print("F1: ")
print(f1_1)
print("ROC: ")
print(y_roc1)

print("ART74 all results:")
print()
y_pred2 = model2.predict(X_test2)
y_recall2 = recall_score(y_test2, y_pred2, average=None)
y_roc2 = roc_auc_score(y_test2, model2.predict_proba(X_test2), multi_class='ovo')
average_precision2 = precision_score(y_test2, y_pred2, average=None)
print("Number of mislabeled points out of a total %d points : %d" % (X_test2.shape[0], (y_test2 != y_pred2).sum()))
score2 = model2.score(X_test2, y_test2)
print("Accuracy: ")
print(score2)
print('Precision score: ')
print(average_precision2)
print("Recall: ")
print(y_recall2)
f1_2 = f1_score(y_test2, y_pred2, average=None)
print("F1: ")
print(f1_2)
print("ROC: ")
print(y_roc2)

print("NN9 all results:")
print()
y_pred3 = model3.predict(X_test3)
y_recall3 = recall_score(y_test3, y_pred3, average=None)
y_roc3 = roc_auc_score(y_test3, model3.predict_proba(X_test3), multi_class='ovo')
average_precision3 = precision_score(y_test3, y_pred3, average=None)
print("Number of mislabeled points out of a total %d points : %d" % (X_test3.shape[0], (y_test3 != y_pred3).sum()))
score3 = model3.score(X_test3, y_test3)
print("Accuracy: ")
print(score3)
print('Precision score: ')
print(average_precision3)
print("Recall: ")
print(y_recall3)
f1_3 = f1_score(y_test3, y_pred3, average=None)
print("F1: ")
print(f1_3)
print("ROC: ")
print(y_roc3)


import matplotlib.pyplot as plt  
from sklearn.metrics import plot_confusion_matrix, jaccard_score

# Plots the Confussion Matrices

plot_confusion_matrix(rfc1, X_test1, y_test1, display_labels=labels, normalize='true')
plt.savefig("confusion_ALL_RFC_ART9_normalized.pdf")
plot_confusion_matrix(rfc1, X_test1, y_test1, display_labels=labels)
plt.savefig("confusion_ALL_RFC_ART9.pdf")

plot_confusion_matrix(rfc2, X_test2, y_test2, display_labels=labels, normalize='true')
plt.savefig("confusion_ALL_RFC_ART74_normalized.pdf")
plot_confusion_matrix(rfc2, X_test2, y_test2, display_labels=labels)
plt.savefig("confusion_ALL_RFC_ART74.pdf")

plot_confusion_matrix(rfc3, X_test3, y_test3, display_labels=labels, normalize='true')
plt.savefig("confusion_ALL_RFC_NN9_normalized.pdf")
plot_confusion_matrix(rfc3, X_test3, y_test3, display_labels=labels)
plt.savefig("confusion_ALL_RFC_NN9.pdf")

plt.show()

print()
print("Jaccard Coefficients:")
jaccard1 = jaccard_score(y_test1, y_pred1, labels=None, pos_label=1, average=None, sample_weight=None)
jaccard2 = jaccard_score(y_test2, y_pred2, labels=None, pos_label=1, average=None, sample_weight=None)
jaccard3 = jaccard_score(y_test3, y_pred3, labels=None, pos_label=1, average=None, sample_weight=None)
print()
print("ART9 Jaccard coefficient for each label:")
print(jaccard1)
print()
print("ART74 Jaccard coefficient for each label:")
print(jaccard2)
print()
print("NN9 Jaccard coefficient for each label:")
print(jaccard3)
