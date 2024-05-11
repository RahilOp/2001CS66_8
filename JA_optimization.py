
from configuration import *
## **IMPORT LIBRARIES**
# importing required libraries
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import time
import os
from itertools import combinations

import pickle
from os import path

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix

from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier

## **Importing Datasets**
filename = dataset_name
featurename="JA"
train_data = pd.read_csv(dataset_path +'/'+ train_dataset_name, sep=',', encoding='utf-8')
test_data = pd.read_csv(dataset_path +'/'+ test_dataset_name, sep=',', encoding='utf-8')
X_train = train_data.drop(columns=['label'],axis=1)
y_train = train_data['label']
X_test = test_data.drop(columns=['label'],axis=1)
y_test = test_data['label']
## **Feature selection Wrapper Methods**
# FS_TOOL
# No.	Abbreviation	                          	Extra Parameters
# *13	hho	Harris Hawk Optimization	       	    No
# *12	ssa	Salp Swarm Algorithm	 	            No
# *11	woa	Whale Optimization Algorithm	    	Yes
# *10	sca	Sine Cosine Algorithm	            	Yes
# *09	ja	Jaya Algorithm	                    	No
# *08	gwo	Grey Wolf Optimizer	                	No
# *07	fpa	Flower Pollination Algorithm	    	Yes
# *06	ba	Bat Algorithm	                    	Yes
# *05	fa	Firefly Algorithm	                	Yes
# *04	cs	Cuckoo Search Algorithm	            	Yes
# *03	de	Differential Evolution	            	Yes
# *02	pso	Particle Swarm Optimization	        	Yes
# *01	ga	Genetic Algorithm	                	Yes
# %1 of train data taken as input to optimization
X_t,temp1,y_t,temp2 = train_test_split(X_train,y_train,train_size=opt_percent, random_state=7)
del temp1,temp2
feature_name = output_path+"/"+filename+"_"+featurename+"_feature.csv"
file = open(feature_name, 'w')
file.write("optimization,execution time of optimzier,no of feature selected,selected feature \n")
file.write(featurename+",")
file.close()
from FS.ja import jfs   # change this to switch algorithm 
# split data into train & validation (70 -- 30)
feat = np.asarray(X_t)
label= np.asarray(y_t)
del X_t,y_t
xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3, stratify=label)
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

# parameter
k    = 5     # k-value in KNN
N    = 10    # number of chromosomes
T    = no_of_opt_iteration   # maximum number of generations
# Extra parameters of listed methods other than population size / number of solutions and maximum number of iterations

# Flower Pollination Algorithm (FPA)
# FPA contains 1 extra parameter

P  = 0.8      # switch probability
opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'P':P}

# perform feature selection
import time
start_time = time.time() 
fmdl = jfs(feat, label, opts)
end_time = time.time()
sf   = fmdl['sf']

# sf is selected_feature
sf = fmdl['sf']
exe_time = end_time - start_time


file = open(feature_name, 'a')
file.write(str(exe_time) +",")
file.write(str(len(sf)) +",")
file.write("\"")
column_headers = list(X_train.columns.values)
for i in sf:
  file.write(column_headers[i]+",")
file.write("\"\n")
file.close()
## **Selection of feature**
feature_df = pd.read_csv(feature_name, sep=',', encoding='utf-8')
selected_feature = feature_df.iat[0, 3]
selected_feature = selected_feature[0:-1]
selected_feature
selected_feature = list(selected_feature.split(","))
selected_feature
X_train = X_train[selected_feature]
X_train
X_test = X_test[selected_feature]
X_test


# Save the filtered test dataset
test_data_filtered = pd.concat([X_test, y_test], axis=1)
dataSetName = output_path+"/"+ filename + "_test_"+ featurename + ".csv"
test_data_filtered.to_csv(dataSetName, index=False)

train_data_filtered = pd.concat([X_train, y_train], axis=1)
dataSetName = output_path+"/"+ filename + "_"+ featurename + "_No_SMOTE.csv"
train_data_filtered.to_csv(dataSetName, index=False)

# Further split the filtered training data and apply SMOTEENN for oversampling



# Apply SMOTEENN to balance the dataset
sm = SMOTE(sampling_strategy='auto', k_neighbors=4, n_jobs=-1)
enn = EditedNearestNeighbours(sampling_strategy='auto', n_neighbors=3, kind_sel='all', n_jobs=-1)
smenn = SMOTEENN(sampling_strategy='auto', smote=sm, enn=enn, n_jobs=-1)
X_train_oversampled, y_train_oversampled = smenn.fit_resample(X_train, y_train)

# Combine balanced features and labels into a single DataFrame
balanced_df = pd.DataFrame(X_train_oversampled, columns=X_train.columns)
balanced_df['label'] = y_train_oversampled

dataSetName = output_path+"/"+ filename + "_"+ featurename + "_SMOTE_ENN.csv"
# Save the entire balanced dataset with SMOTEENN applied
balanced_df.to_csv(dataSetName, index=False)

# Optionally, apply SMOTETomek for another resampling technique
smt = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smt.fit_resample(X_train, y_train)

# Combine balanced features and labels into a single DataFrame
balanced_df_ipf = pd.DataFrame(X_train_resampled, columns=X_train.columns)
balanced_df_ipf['label'] = y_train_resampled
balanced_df_ipf.reset_index(drop=True, inplace=True)

dataSetName = output_path+"/"+ filename + "_"+ featurename + "_SMOTE_IPF.csv"

# Save the balanced dataset with SMOTETomek applied
balanced_df_ipf.to_csv(dataSetName, index=False)


import smote_variants as sv
import pandas as pd


algorithms = [
    "Safe_Level_SMOTE", "RWO_sampling", "ROSE",
    "SMOTE_OUT", "Selected_SMOTE", "SN_SMOTE"
]

for algorithm in algorithms:
    try:
        oversampler = sv.MulticlassOversampling(oversampler=algorithm,
                                                oversampler_params={'random_state': 5})

        # X_samp and y_samp contain the oversampled dataset
        X_samp, y_samp = oversampler.sample(X_train, y_train)

        # Create DataFrame from oversampled data
        oversampled_df = pd.DataFrame(data=X_samp, columns=[f'feature_{i}' for i in range(X_samp.shape[1])])
        oversampled_df['label'] = y_samp
        oversampled_df.reset_index(drop=True, inplace=True)  # Reset the index

        # Define output CSV file name
        dataSetName = f"{output_path}/{filename}_{featurename}_{algorithm}.csv"
        oversampled_df.to_csv(dataSetName, index=False)

        print(f'Oversampled dataset saved to {dataSetName}')
    except Exception as e:
        print(f"Error processing {algorithm}: {str(e)}")
        continue