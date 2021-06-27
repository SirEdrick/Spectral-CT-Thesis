# -*- coding: utf-8 -*-
"""
Created on Fri Jul 4 10:32:21 2021

@author: mbust, bruun
"""

import itertools
import pandas as pd
import numpy as np

#--------------------setup the data-----------------------
#-labels
#-df type file

#datapath_head = "G:/StudieBackup/Test_dataset/Sample_06062018_Fluids"
#datapath_head = "G:/StudieBackup/Test_dataset/Sample_06062018_Fruits"
#datapath_head = "G:/StudieBackup/Test_dataset/Sample_06062018_NonThreat"
datapath_head = "G:/StudieBackup/Test_dataset/Sample_06062018_Threat"
#datapath_head = "G:/StudieBackup/Test_dataset/Sample_23012018"
#datapath_head = "G:/StudieBackup/Test_dataset/Sample_24012018"

#labels = ["acetone","h2o","h2o2","nitric_acid","olive_oil","whiskey"]
#labels = ["grapes","apple","tropicalJuice","citrusJuice"]
#labels = ["Nivea_sun_lotion_pf50+", "Olive_oil", "Toothpaste","h2o","Cien_hand_creme"]
labels = ["acetone","C4","h2o2","methanol"]
#labels = ["h2o", "h2o2", "whiskey","hand_cream","toothaste","C4","aluminium_rod"]
#labels = ["aluminium_rod", "C4", "hand_cream","h2o2","toothaste","h2o","whiskey"]

#-Labels Load for each type

# ART9

file1 = datapath_head + "/processed/segmentedART9/"+'labels_all.txt'

labels_all1 = pd.read_csv(file1, delimiter = "\t", header=None)
flat_labels1 = labels_all1.to_numpy().flatten()
label_names_set1 = set(flat_labels1)

label_ids1 = [labels.index(x) for x in flat_labels1]
label_ids_set1 = set(label_ids1)

# ART74

file2 = datapath_head + "/processed/segmentedART74/"+'labels_all.txt'

labels_all2 = pd.read_csv(file2, delimiter = "\t", header=None)
flat_labels2 = labels_all2.to_numpy().flatten()
label_names_set2 = set(flat_labels2)

label_ids2 = [labels.index(x) for x in flat_labels2]

#-LAC data1

print("Loading LAC data ART9...")
data_file1 = datapath_head + "/processed/segmentedART9/"+'LAC_all.csv'
        
columns1 = list(np.arange(1,33))

data_allART9 = pd.read_csv(data_file1, header=None)
data_allART9.columns = columns1
data_allART9['ids'] = label_ids1

print("LAC data loaded. Number of voxels: {}".format(len(data_allART9)))

# -LAC data2

print("Loading LAC data ART74...")
data_file2 = datapath_head + "/processed/segmentedART74/" + 'LAC_all.csv'

columns2 = list(np.arange(1, 33))

data_allART74 = pd.read_csv(data_file2, header=None)
data_allART74.columns = columns2
data_allART74['ids'] = label_ids2

print("LAC data loaded. Number of voxels: {}".format(len(data_allART74)))

print()
print("Creating sklearning model...")

X1, y1 = data_allART9.iloc[:,:-1],label_ids1

print()
print("Printing 9 Projection Data")
unique_numbers1 = list(set(y1))
X1_np = X1.to_numpy()
x_1 = np.arange(0,32)

import matplotlib.pyplot as plt
for seg_id in itertools.islice(itertools.count(), 0, len(unique_numbers1)):
            print("label_id is:")
            print(seg_id+1)
            print(len(unique_numbers1))
            idx1 = np.where(np.array(y1)==seg_id)
            plt.plot(x_1,X1_np[list(idx1[0]),:].transpose())
            plt.show()

X2, y2 = data_allART74.iloc[:,:-1],label_ids2

print()
print("Printing 74 Projection Data")
unique_numbers2 = list(set(y2))
X2_np = X2.to_numpy()
x_2 = np.arange(0,32)

import matplotlib.pyplot as plt
for seg_id in itertools.islice(itertools.count(), 0, len(unique_numbers2)):
            print("label_id is:")
            print(seg_id+1)
            print(len(unique_numbers2))
            idx2 = np.where(np.array(y2)==seg_id)
            plt.plot(x_2,X2_np[list(idx2[0]),:].transpose())
            plt.show()