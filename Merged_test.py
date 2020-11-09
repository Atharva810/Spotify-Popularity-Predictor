#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold
from collections import defaultdict
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
import statistics
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from collections import Counter


# In[ ]:
threshold=50
def detect_outliers(df, features):
    outlier_indices = []

    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c], 25)
        # 3rd quartile
        Q3 = np.percentile(df[c], 75)
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step)
                              | (df[c] > Q3 + outlier_step)].index  # filtre
        # store indeces
        # The extend() extends the list by adding all items of a list (passed as an argument) to the end.
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(
            i for i, v in outlier_indices.items() if v > 2)

    return multiple_outliers


filenames = os.listdir('./searched_songs/')
filepath = "./searched_songs/"


# In[ ]:


features = ["acousticness", "danceability", "energy", "instrumentalness", "key", "liveness",
                "mode", "loudness", "speechiness", "tempo", "valence","length","time_signature"]


# In[ ]:
outlier_features = ["popularity", "length", "danceability", "energy",
                    "instrumentalness", "liveness", "loudness", "speechiness", "valence"]

test_df = pd.DataFrame() 

for file in filenames:
    whole_dataset = pd.read_csv(filepath+file, encoding='utf-8')
    # whole_dataset = whole_dataset.drop(detect_outliers(whole_dataset, outlier_features), axis=0).reset_index(drop=True)
    test_len = int(len(whole_dataset)*0.1)
    test = whole_dataset[:test_len]
    print(len(test))
    print(len(whole_dataset))
    test_df = pd.concat([test,test_df])
    print(len(test_df))
    test_df.drop_duplicates(subset="trackid",inplace=True)
    print(len(test_df))


test_df["thresholded_popularity"] = [
    1 if i >= threshold else 0 for i in test_df.popularity]

def change_type(var):
    whole_dataset[var] = whole_dataset[var].astype(int)

column = ["key", "length", "mode"]

for file in filenames:
    print(file)
    whole_dataset = pd.read_csv(filepath+file, encoding='utf-8')

    # In[ ]:


    whole_dataset = whole_dataset.loc[:, ~whole_dataset.columns.str.match('Unnamed')]
    # whole_dataset = whole_dataset.drop(detect_outliers(whole_dataset, outlier_features), axis=0).reset_index(drop=True)

    # In[ ]:


    # print(len(whole_dataset))

    # In[ ]:



    # whole_dataset.loc[detect_outliers(whole_dataset,outlier_features)]
    # print(len(whole_dataset.loc[detect_outliers(whole_dataset,outlier_features)]))

    # In[ ]:


    # print(len(whole_dataset))


    # In[ ]:


    whole_dataset["thresholded_popularity"]= [ 1 if i>= threshold else 0 for i in whole_dataset.popularity ]
    whole_dataset["thresholded_popularity"].value_counts()
    print(whole_dataset.thresholded_popularity.value_counts())


    # In[ ]:





    # In[ ]:


    for i in column:
        change_type(i)


    # In[ ]:

    train_len=int(len(whole_dataset)*0.1)
    dataset = whole_dataset[train_len:]

    # x,y = whole_dataset[features] , whole_dataset.loc[:,'thresholded_popularity']
    # x = (x - np.min(x))/(np.max(x)-np.min(x)).values
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)
    x_train = dataset[features]
    y_train = dataset["thresholded_popularity"]

    

    x_test = test_df[features]
    y_test = test_df["thresholded_popularity"]

    # y=y.astype(int)
    y_train= y_train.astype(int)
    y_test= y_test.astype(int)
    # x_train.info()


    # In[ ]:


    # whole_dataset.columns[whole_dataset.isnull().any()]

    # song_data.columns[song_data.isnull().any()]
    # whole_dataset.isnull().sum()

    rf = RandomForestClassifier(n_estimators=150, random_state=3)
    rf.fit(x_train, y_train)
    print("Train ccuracy of random forest", rf.score(x_train, y_train))
    print("Test accuracy of random forest", rf.score(x_test, y_test))
    RandomForestClassifier_score = rf.score(x_test, y_test)
    y_pred = rf.predict(x_test)
    t_true = y_test

    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix: \n', cm)

    cr = classification_report(y_test, y_pred, output_dict=False)
    print('Classification report: \n', cr
        )

