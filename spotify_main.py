import numpy as np 
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold
# from xgboost import XGBClassifier

from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

filenames = os.listdir('./filtered_dataset/')
filepath = "./filtered_dataset/"

test = pd.read_csv('merged_without_duplicates.csv')
features = ["acousticness", "danceability", "energy", "instrumentalness", "key", "liveness", 
                "mode", "loudness", "speechiness", "tempo", "valence"]

test_cols = test.sample(frac = 0.2,random_state = 420)
cols = test_cols["trackid"]
result = pd.DataFrame(columns=cols)
print(result.head())
for file in filenames:
    #dataframe = pd.read_csv('filtered_dataset/us.csv')
    print(file)
    dataframe = pd.read_csv(filepath+file,encoding='utf-8')
    # dataframe.head()
    # print(test.head())
    threshold = np.mean(dataframe.popularity)
    sorted1 = dataframe.nlargest(int(len(dataframe.index)*0.25),'popularity')
    # sorted1.tail(5)
    threshold = list(sorted1.popularity)[-1]
    print(threshold)

    dataframe.loc[dataframe['popularity'] < threshold, 'popularity'] = 0 
    dataframe.loc[dataframe['popularity'] >= threshold, 'popularity'] = 1
    # dataframe.loc[dataframe['popularity'] == 1]
    test.loc[test['popularity'] < threshold, 'popularity'] = 0 
    test.loc[test['popularity'] >= threshold, 'popularity'] = 1

    

    training = dataframe.sample(frac = 1.0,random_state = 420)
    X_train = training[features]
    y_train = training['popularity']
    
    testing = test.sample(frac = 0.2,random_state = 420)
    X_test = testing[features]
    y_test = testing['popularity']
    #X_test = dataframe.drop(training.index)[features]
    # len(testing)
    # print(len(testing))
    # print(test.shape)

    RFC_Model = RandomForestClassifier()
    RFC_Model.fit(X_train, y_train)
    RFC_Predict = RFC_Model.predict(X_test)
    print(RFC_Predict)
    RFC_Accuracy = accuracy_score(y_test, RFC_Predict)
    print("Accuracy: " + str(RFC_Accuracy))
    result.append(RFC_Predict)

print(result.head())
# print(RFC_Predict)