import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
# from xgboost import XGBClassifier

from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

filenames = os.listdir('./filtered_dataset/')
filepath = "./filtered_dataset/"

test = pd.read_csv('merged_without_duplicates.csv')
features = ["acousticness", "danceability", "energy", "instrumentalness", "key", "liveness",
                "mode", "loudness", "speechiness", "tempo", "valence"]

testing = test.sample(frac = 1, random_state = 420)

# test_cols = test.sample(frac = 0.1,random_state = 420)
cols = pd.DataFrame(testing["trackid"], index=None)

result = pd.DataFrame(columns=cols)

country_list = []
for file in filenames:
    country_list.append(file.split(".")[0])

popular_song_dict = defaultdict(list)

lowest_accuracy = 100
i=0
for file in filenames:
    testing_current_country = pd.DataFrame(testing)
    # print(testing_current_country.head())
    #dataframe = pd.read_csv('filtered_dataset/us.csv')
    print(file)
    dataframe = pd.read_csv(filepath+file,encoding='utf-8')
    # dataframe.head()
    # print(test.head())
    # threshold = np.mean(dataframe.popularity)
    sorted1 = dataframe.nlargest(int(len(dataframe.index)*0.25),'popularity')
    # sorted1.tail(5)
    threshold = list(sorted1.popularity)[-1]
    # print(threshold)

    dataframe.loc[dataframe['popularity'] < threshold, 'popularity'] = 0
    dataframe.loc[dataframe['popularity'] >= threshold, 'popularity'] = 1
    # dataframe.loc[dataframe['popularity'] == 1]

    testing_current_country.loc[testing_current_country['popularity']
                                < threshold, 'popularity'] = 0
    testing_current_country.loc[testing_current_country['popularity']
                                >= threshold, 'popularity'] = 1

    testing_current_country.to_csv(r'./testing_current_countries/'+file)
    i+=1

    # popular_songs = dataframe.loc[dataframe["popularity"] == 1]
    popular_songs = testing_current_country.loc[testing_current_country["popularity"] == 1]
    # print(len(popular_songs))
    print(len(popular_songs))
    # print(len(dataframe))
    # print(len(testing))
    for idx,row in popular_songs.iterrows():
        popular_song_dict[row["trackid"]].append(file.split(".")[0])
    # print(popular_songs.head())
    # print(popular_song_dict)
    # exit()
    training = dataframe.sample(frac = 1.0,random_state = 420)
    X_train = training[features]
    y_train = training['popularity']

    X_test = testing_current_country[features]
    y_test = testing_current_country['popularity']

    #X_test = dataframe.drop(training.index)[features]
    # len(testing)
    # print(len(testing))
    # print(test.shape)

    RFC_Model = RandomForestClassifier()
    RFC_Model.fit(X_train, y_train)
    RFC_Predict = RFC_Model.predict(X_test)
    RFC_Predict = list(RFC_Predict)
    # print(RFC_Predict)
    RFC_Accuracy = accuracy_score(y_test, RFC_Predict)
    print("Accuracy: " + str(RFC_Accuracy))
    lowest_accuracy = min(lowest_accuracy, RFC_Accuracy)

    result.loc[len(result)] = RFC_Predict
    # result.loc[len(result)]["index"] = file.split(".")[0]
    # cm = confusion_matrix(y_test,RFC_Predict)
    # print('Confusion matrix: \n',cm)
    # print('Classification report: \n',classification_report(y_test,RFC_Predict))
    # TP = cm[1, 1]
    # TN = cm[0, 0]
    # FP = cm[0, 1]
    # FN = cm[1, 0]
    #
    # print((TP + TN) / float(TP + TN + FP + FN))z

result["Countries"] = country_list
print(result.head())
print(lowest_accuracy)
# print(RFC_Predict)
# d = defaultdict(list, {k: [] for k in ('a', 'b', 'c')})
dict_of_countries = defaultdict(list, {t: [] for t in cols["trackid"]})
# print(cols)
for idx, col in enumerate(cols["trackid"]):
    track = result.iloc[:,idx]
    countries = result.Countries
    for i,t in enumerate(track):
        if t == 1:
            dict_of_countries[col].append(countries[i])
        # else:
        #     dict_of_countries[col].append("")
        # print(type(t))
# for idx,row in result.iterrows():
#     print(row)
#     for idx,col in enumerate(cols["trackid"]):
#         # print(col)
#         if row.iloc[:,idx] == 1:
#             dict_of_countries[col].append(row.Countries)
print(len(dict_of_countries))
print(len(cols))
# print(dict_of_countries)
print(popular_song_dict)
