# random forest for feature importance on a classification problem
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
# define dataset

filenames = os.listdir('./searched_songs/')
filepath = "./searched_songs/"

features = ["acousticness", "danceability", "energy", "instrumentalness", "key", "liveness",
            "mode", "loudness", "speechiness", "tempo", "valence", "length", "time_signature"]

top3features = {}

threshold_dict = {"ar": 70, "bo": 70, "br": 70, "cl": 70, "co": 70, "cz": 70, "dk": 70, "do": 70, "fi": 70, "fr": 70,
                  "gt": 70, "id": 70, "in": 70, "it": 70, "pa": 70, "pe": 70, "ph": 70, "pl": 70, "ru": 70, "tr": 70, "za": 70}


# function for creating a feature importance dataframe
def imp_df(column_names, importances):
    df = pd.DataFrame({'feature': column_names,
                       'feature_importance': importances}) \
        .sort_values('feature_importance', ascending=False) \
        .reset_index(drop=True)
    return df

# plotting a feature importance dataframe (horizontal barchart)


def var_imp_plot(imp_df, title):
    imp_df.columns = ['feature', 'feature_importance']
    sns.barplot(x='feature_importance', y='feature', data=imp_df, orient='h', color='royalblue') \
       .set_title(title, fontsize=20)

features_dict = {}

for file in filenames:
    print(file)
    country = file.split(".")[0]
    # country = file.split(".")[0]
    dataframe = pd.read_csv(filepath+file, encoding='utf-8')

    threshold = 50

    if country in threshold_dict:
        threshold = threshold_dict[country]

    dataframe.loc[dataframe['popularity'] <
                threshold, 'thresholded_popularity'] = 0
    dataframe.loc[dataframe['popularity'] >=
                threshold, 'thresholded_popularity'] = 1

    training = dataframe.sample(frac=1, random_state=420)
    X_train = training[features]
    Y_train = training['thresholded_popularity']
    # X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
    # X_train, X_valid, Y_train, Y_valid = train_test_split(
    #     X_train, Y_train, test_size=0.2, random_state=420)
    # define the model
    model = RandomForestClassifier()
    # fit the model
    model.fit(X_train, Y_train)
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    # print(list(zip(X_train.columns, importance)))
    base_imp = imp_df(X_train.columns, importance)
    country_features= []
    top3 = []
    for idx,row in base_imp.iterrows():
        if idx==3:
            break
        top3.append(row.feature)
        country_features.append([row.feature, row.feature_importance])

    top3 = str(top3)
    if top3 in top3features:
        top3features[top3].append(country)
    else:
        top3features[top3] = [country]

    features_dict[country] = country_features

    # print(base_imp)
    var_imp_plot(base_imp, file)
    # plt.show()
    # for feature,imp in zip(X_train.columns, importance):
    #     print('Feature: %0d, Score: %.5f' % (i,v))
    # # plot feature importance
    # pyplot.bar([x for x in range(len(importance))], importance)
    # pyplot.show()

with open("countrywise_features.json", "w", encoding='utf-8') as f:
    json.dump(features_dict, f)

with open("countrywise_top3_features.json", "w", encoding='utf-8') as f:
    json.dump(top3features, f)
