import numpy as np 
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
import statistics
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score 
from sklearn.model_selection import train_test_split
import json
import pdb

filenames = os.listdir('./searched_songs/')
filepath = "./searched_songs/"

features = ["acousticness", "danceability", "energy", "instrumentalness", "key", "liveness",
            "mode", "loudness", "speechiness", "tempo", "valence", "length", "time_signature"]

threshold_dict = {"ar": 70, "bo": 70, "br": 70, "cl": 70, "co": 70, "cz": 70, "dk": 70, "do": 70, "fi": 70, "fr": 70,
                  "gt": 70, "id": 70, "in": 70, "it": 70, "pa": 70, "pe": 70, "ph": 70, "pl": 70, "ru": 70, "tr": 70, "za": 70}
accuracy_dict = {}
ones_zeros_dict = {}


lowest_accuracy = 100
for file in filenames:
    print(file)
    dataframe = pd.read_csv(filepath+file,encoding='utf-8')    
    country = file.split(".")[0]
    threshold = 50
    if country in threshold_dict:
        threshold = threshold_dict[country]

    dataframe.loc[dataframe['popularity'] < threshold, 'thresholded_popularity'] = 0 
    dataframe.loc[dataframe['popularity'] >=
                  threshold, 'thresholded_popularity'] = 1
    division = dataframe.thresholded_popularity.value_counts()
    ones = int(division.values[0])
    zeros = int(division.values[1])
    ones_zeros_dict[country] = [zeros,ones]
    print(division)

    # pdb.set_trace()

    training = dataframe.sample(frac = 1,random_state = 420)
    X_train = training[features]
    Y_train = training['thresholded_popularity']
    
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=420)
    RFC_Model = RandomForestClassifier(n_estimators=300)
    RFC_Model.fit(X_train, Y_train)
    RFC_Predict = RFC_Model.predict(X_valid)
    RFC_Accuracy = accuracy_score(Y_valid, RFC_Predict)
    print("RF Accuracy: " + str(RFC_Accuracy))
    accuracy_dict[country] = RFC_Accuracy
    lowest_accuracy = min(lowest_accuracy, RFC_Accuracy)
    cm = confusion_matrix(Y_valid, RFC_Predict)
    print('Confusion matrix: \n', cm)
    print('Classification report: \n',
          classification_report(Y_valid, RFC_Predict))
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
print(f"Lowest Accuracy: {lowest_accuracy}")
with open("countrywise_accuracy.json", "w", encoding='utf-8') as f:
    json.dump(accuracy_dict, f)

with open("countrywise_ones_zeros.json", "w", encoding='utf-8') as f:
    json.dump(ones_zeros_dict, f)


