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
import json

filenames = os.listdir('./searched_songs/')
filepath = "./searched_songs/"

# features = ["acousticness", "danceability", "energy", "instrumentalness", "key", "liveness", 
#                 "mode", "loudness", "speechiness", "tempo", "valence"]

features = ["acousticness", "danceability", "energy", "instrumentalness", "key", "liveness",
            "mode", "loudness", "speechiness", "tempo", "valence", "length", "time_signature"]

# accuracy_dict = {}

threshold_dict = {}
lowest_accuracy = 100

for file in filenames:
    print(file)
    country = file.split(".")[0]
    dataframe = pd.read_csv(filepath+file,encoding='utf-8')
    
    # max_precision = -float("inf")
    # max_accuracy = -float("inf")
    # threshold_for_max_values = 0
    threshold_dict = {}
    for threshold in range(30,71):
    
        dataframe.loc[dataframe['popularity'] < threshold, 'thresholded_popularity'] = 0 
        dataframe.loc[dataframe['popularity'] >=
                    threshold, 'thresholded_popularity'] = 1

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
        # accuracy_dict[file.split(".")[0]] = RFC_Accuracy
        lowest_accuracy = min(lowest_accuracy, RFC_Accuracy)

        cm = confusion_matrix(Y_valid, RFC_Predict)
        print('Confusion matrix: \n', cm)
        cr = classification_report(Y_valid, RFC_Predict, output_dict=True)
        print('Classification report: \n',cr)


        del(cr["accuracy"])
        precision_sum = 0
        i = 0
        for key, value in cr.items():
            if i == 2:
                break
            precision_sum += value["precision"]
            i += 1
        # precision_dict[threshold] = precision_sum
        # if precision_sum > max_sum:
        #     max_sum = precision_sum
        #     best_threshold = threshold

        TP = cm[1, 1]
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]

        threshold_dict[threshold] = [RFC_Accuracy, precision_sum]
    with open("countrywise_analysis\\"+file, "w", encoding='utf-8') as f:
        json.dump(threshold_dict, f)

print(f"Lowest Accuracy: {lowest_accuracy}")
# with open("countrywise_accuracy.json", "w", encoding='utf-8') as f:
#     json.dump(accuracy_dict, f)


