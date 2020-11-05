import numpy as np 
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
import matplotlib.pyplot as plt
import seaborn as sns

filenames = os.listdir('./filtered_dataset/')
filepath = "./filtered_dataset/"

features = ["acousticness", "danceability", "energy", "instrumentalness", "key", "liveness", 
                "mode", "loudness", "speechiness", "tempo", "valence"]

sum_rfaccuracy=0
count=0
lowest_accuracy = 100
whole_dataset = pd.read_csv("us.csv", encoding='utf-8')
dataframe = pd.read_csv(filepath+"us.csv", encoding='utf-8')
merged = pd.read_csv("merged_without_duplicates.csv", encoding='utf-8')
print(len(dataframe))

print(len(whole_dataset))
whole_dataset = pd.concat([whole_dataset, dataframe])
whole_dataset = whole_dataset.drop_duplicates(subset=["trackid"])
# whole_dataset = pd.DataFrame(whole_dataset, index=range(len(dataframe)))
whole_dataset["popularity"] = 0
for idx, row in dataframe.iterrows():
    whole_dataset.loc[whole_dataset["trackid"]==row.trackid,"popularity"] = 1


f, ax = plt.subplots(figsize=(12, 12))
mask = np.zeros_like(dataframe.corr())
mask[np.triu_indices_from(mask)] = True
sns.heatmap(dataframe.corr(), annot=True, linewidths=0.4,
            linecolor="white", fmt='.1f', ax=ax, cmap="Blues", mask=mask)
plt.show()




f, axes = plt.subplots(3, 5, figsize=(12, 12))
# sns.distplot(whole_dataset["song_duration_ms"], color="teal", ax=axes[0, 0])
sns.distplot(whole_dataset["instrumentalness"], color="teal", ax=axes[0, 1])
# sns.distplot(whole_dataset["acousticness"], color="teal", ax=axes[0, 2])
# sns.distplot(whole_dataset["danceability"], color="teal", ax=axes[0, 3])
# sns.distplot(whole_dataset["energy"], color="teal", ax=axes[0, 4])
# # sns.distplot(whole_dataset["song_popularity"], color="teal", ax=axes[1, 0])
# sns.distplot(whole_dataset["key"], color="teal", ax=axes[1, 1])
# sns.distplot(whole_dataset["liveness"], color="teal", ax=axes[1, 2])
# sns.distplot(whole_dataset["loudness"], color="teal", ax=axes[1, 3])
# sns.distplot(whole_dataset["mode"], color="teal", ax=axes[1, 4])
# sns.distplot(whole_dataset["tempo"], color="teal", ax=axes[2, 0])
# sns.distplot(whole_dataset["speechiness"], color="teal", ax=axes[2, 1])
# # sns.distplot(whole_dataset["time_signature"], color="teal", ax=axes[2, 2])
# sns.distplot(whole_dataset["valence"], color="teal", ax=axes[2, 3])
f.delaxes(axes[2][4])
plt.show()



# threshold = whole_dataset["popularity"].quantile(0.60)
# print(threshold)
# whole_dataset.loc[whole_dataset['popularity'] < threshold, 'popularity'] = 0
# whole_dataset.loc[whole_dataset['popularity'] >= threshold, 'popularity'] = 1
print(whole_dataset.popularity.value_counts())
training = whole_dataset.sample(frac=1, random_state=420)
X_train = training[features]
y_train = training['popularity']

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=420)
RFC_Model = RandomForestClassifier(n_estimators=300)
# RFC_Model = RandomForestClassifier(n_estimators=1800, min_samples_split=5, min_samples_leaf=2, max_features='auto', max_depth=20, bootstrap=False)
RFC_Model.fit(X_train, y_train)
RFC_Predict = RFC_Model.predict(X_valid)
RFC_Accuracy = accuracy_score(y_valid, RFC_Predict)
sum_rfaccuracy += RFC_Accuracy
count += 1
print("RF Accuracy: " + str(RFC_Accuracy))
lowest_accuracy = min(lowest_accuracy, RFC_Accuracy)

cm = confusion_matrix(y_valid, RFC_Predict)
print('Confusion matrix: \n', cm)
print('Classification report: \n',
        classification_report(y_valid, RFC_Predict))
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))

print(f"Lowest Accuracy: {lowest_accuracy}")
print(f"Average RF accuracy: {sum_rfaccuracy/count}")


# for file in filenames:
#     print(file)
#     dataframe = pd.read_csv(filepath+file,encoding='utf-8')
#     threshold = dataframe["popularity"].quantile(0.8)
#     print(threshold)
#     dataframe.loc[dataframe['popularity'] < threshold, 'popularity'] = 0 
#     dataframe.loc[dataframe['popularity'] >= threshold, 'popularity'] = 1

#     training = dataframe.sample(frac = 1,random_state = 420)
#     X_train = training[features]
#     y_train = training['popularity']
    
#     X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = 420)    

#     RFC_Model = RandomForestClassifier(n_estimators=300)
#     RFC_Model.fit(X_train, y_train)
#     RFC_Predict = RFC_Model.predict(X_valid)
#     RFC_Accuracy = accuracy_score(y_valid, RFC_Predict)
#     sum_rfaccuracy += RFC_Accuracy
#     count+=1
#     print("RF Accuracy: " + str(RFC_Accuracy))
#     lowest_accuracy = min(lowest_accuracy, RFC_Accuracy)

#     cm = confusion_matrix(y_valid, RFC_Predict)
#     print('Confusion matrix: \n', cm)
#     print('Classification report: \n',
#           classification_report(y_valid, RFC_Predict))
#     TP = cm[1, 1]
#     TN = cm[0, 0]
#     FP = cm[0, 1]
#     FN = cm[1, 0]

#     print((TP + TN) / float(TP + TN + FP + FN))

# print(f"Lowest Accuracy: {lowest_accuracy}")
# print(f"Average RF accuracy: {sum_rfaccuracy/count}")

