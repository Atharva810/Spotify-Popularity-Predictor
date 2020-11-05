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
# from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import statistics
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# filenames = os.listdir('./filtered_dataset/')
filenames = os.listdir('./dataset/')
# filepath = "./filtered_dataset/"
filepath = "./dataset/"

# test = pd.read_csv('merged_without_duplicates.csv')
features = ["acousticness", "danceability", "energy", "instrumentalness", "key", "liveness", 
                "mode", "loudness", "speechiness", "tempo", "valence"]

# testing = test.sample(frac = 0.1, random_state = 420)

# test_cols = test.sample(frac = 0.1,random_state = 420)
# cols = pd.DataFrame(testing["trackid"], index=None)

# result = pd.DataFrame(columns=cols)

# country_list = []
# for file in filenames:
#     country_list.append(file.split(".")[0])

popular_song_dict = defaultdict(list)
# test = pd.read_csv('merged_without_duplicates.csv')
# sorted1 = test.nlargest(int(len(test.index)*0.26), 'popularity')
# threshold = np.mean(test["popularity"])
# threshold = list(sorted1.popularity)[-1]
# print(threshold)
sum_rfaccuracy=0
# sum_dtaccuracy=0
# sum_xgbaccuracy=0
# sum_knnaccuracy=0
# sum_lraccuracy = 0
count=0
lowest_accuracy = 100
for file in filenames:
    # testing_current_country = pd.DataFrame(testing)
    #dataframe = pd.read_csv('filtered_dataset/us.csv')
    print(file)
    dataframe = pd.read_csv(filepath+file,encoding='utf-8')
    # print(dataframe.describe())
    # dataframe.head()
    # print(test.head())
    # threshold = np.mean(dataframe.popularity)
    # sorted1 = dataframe.nlargest(int(len(dataframe.index)*0.30),'popularity')
    # # sorted1.tail(5)
    # threshold = list(sorted1.popularity)[-1]
    # print(threshold)
    threshold = dataframe["popularity"].quantile(0.8)
    print(threshold)
    dataframe.loc[dataframe['popularity'] < threshold, 'popularity'] = 0 
    dataframe.loc[dataframe['popularity'] >= threshold, 'popularity'] = 1
    # dataframe.loc[dataframe['popularity'] == 1]
    
    # testing_current_country.loc[testing_current_country['popularity']
    #                             < threshold, 'popularity'] = 0
    # testing_current_country.loc[testing_current_country['popularity']
    #                             >= threshold, 'popularity'] = 1

    # popular_songs = dataframe.loc[dataframe["popularity"] == 1]
    # popular_songs = testing_current_country.loc[testing_current_country["popularity"] == 1]
    # print(len(popular_songs))
    # print(len(popular_songs))
    # print(len(dataframe))
    # print(len(testing))
    # for idx,row in popular_songs.iterrows():
    #     popular_song_dict[row["trackid"]].append(file.split(".")[0])
    # print(popular_songs.head())
    # print(popular_song_dict)
    # exit()
    training = dataframe.sample(frac = 1,random_state = 420)
    X_train = training[features]
    y_train = training['popularity']
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = 420)    
    
    # X_test = testing_current_country[features]
    # y_test = testing_current_country['popularity']
    
    #X_test = dataframe.drop(training.index)[features]
    # len(testing)
    # print(len(testing))
    # print(test.shape)



    # pprint(RFC_Model.get_params())
    # n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # max_features = ['auto', 'sqrt','log2']
    # max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    # max_depth.append(None)
    # min_samples_split = [2, 5, 10]
    # min_samples_leaf = [1, 2, 4]
    # bootstrap = [True, False]
    # random_grid = {'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth,
    #                'min_samples_split': min_samples_split,
    #                'min_samples_leaf': min_samples_leaf,
    #                'bootstrap': bootstrap}
    # pprint(random_grid)
    # RFC_Model = RandomForestClassifier(n_estimators=1800, min_samples_split= 5, min_samples_leaf=2, max_features='auto', max_depth= 20, bootstrap=False)
    RFC_Model = RandomForestClassifier(n_estimators=300)
    # RFC_Model = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                #    n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
    RFC_Model.fit(X_train, y_train)
    RFC_Predict = RFC_Model.predict(X_valid)
    RFC_Accuracy = accuracy_score(y_valid, RFC_Predict)
    sum_rfaccuracy += RFC_Accuracy
    count+=1
    print("RF Accuracy: " + str(RFC_Accuracy))
    lowest_accuracy = min(lowest_accuracy, RFC_Accuracy)
    # print(RFC_Model.best_params_)
    # exit()
    # DT_Model = DecisionTreeClassifier()
    # DT_Model.fit(X_train, y_train)
    # DT_Predict = DT_Model.predict(X_valid)
    # DT_Accuracy = accuracy_score(y_valid, DT_Predict)
    # sum_dtaccuracy += DT_Accuracy
    # print("DT Accuracy: " + str(DT_Accuracy))


    # XGB_Model = XGBClassifier(objective="binary:logistic",
    #                         n_estimators=10, seed=123)
    # XGB_Model.fit(X_train, y_train)
    # XGB_Predict = XGB_Model.predict(X_valid)
    # XGB_Accuracy = accuracy_score(y_valid, XGB_Predict)
    # sum_xgbaccuracy += XGB_Accuracy
    # print("XGB Accuracy: " + str(XGB_Accuracy))


    # LR_Model = LogisticRegression()
    # LR_Model.fit(X_train, y_train)
    # LR_Predict = LR_Model.predict(X_valid)
    # LR_Accuracy = accuracy_score(y_valid, LR_Predict)
    # sum_lraccuracy += LR_Accuracy
    # print("LR Accuracy: " + str(LR_Accuracy))


    # KNN_Model = KNeighborsClassifier()
    # KNN_Model.fit(X_train, y_train)
    # KNN_Predict = KNN_Model.predict(X_valid)
    # KNN_Accuracy = accuracy_score(y_valid, KNN_Predict)
    # sum_knnaccuracy += KNN_Accuracy
    # print("Accuracy: " + str(KNN_Accuracy))

    cm = confusion_matrix(y_valid, RFC_Predict)
    print('Confusion matrix: \n', cm)
    print('Classification report: \n',
          classification_report(y_valid, RFC_Predict))
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    print((TP + TN) / float(TP + TN + FP + FN))

    # result.loc[len(result)] = RFC_Predict
    # result.loc[len(result)]["index"] = file.split(".")[0]

# result["Countries"] = country_list
# print(result.head())
print(f"Lowest Accuracy: {lowest_accuracy}")
print(f"Average RF accuracy: {sum_rfaccuracy/count}")
# print(f"Average DT accuracy: {sum_dtaccuracy/count}")
# print(f"Average XGB accuracy: {sum_xgbaccuracy/count}")
# print(f"Average LR accuracy: {sum_lraccuracy/count}")
# print(f"Average KNN accuracy: {sum_knnaccuracy/count}")
# print(RFC_Predict)
# d = defaultdict(list, {k: [] for k in ('a', 'b', 'c')})
# dict_of_countries = defaultdict(list, {t: [] for t in cols["trackid"]})
# # print(cols)
# for idx, col in enumerate(cols["trackid"]):
#     track = result.iloc[:,idx]
#     countries = result.Countries
#     for i,t in enumerate(track):
#         if t == 1:
#             dict_of_countries[col].append(countries[i])
        # else:
        #     dict_of_countries[col].append("")
        # print(type(t))
# for idx,row in result.iterrows():
#     print(row)
#     for idx,col in enumerate(cols["trackid"]):
#         # print(col)
#         if row.iloc[:,idx] == 1:
#             dict_of_countries[col].append(row.Countries)
# print(len(dict_of_countries))
# print(len(cols))
# # print(dict_of_countries)
# print(popular_song_dict)

