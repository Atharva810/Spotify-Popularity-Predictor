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
import json 

filenames = os.listdir('./searched_songs/')
filepath = "./searched_songs/"


# In[46]:

thresholds = {}

features = ["acousticness", "danceability", "energy", "instrumentalness", "key", "liveness",
                "mode", "loudness", "speechiness", "tempo", "valence","length","time_signature"]


# In[ ]:


# threshold=0
best_threshold =0
threshold=0
precision_dict = {}
max_sum=-float("inf")
# for threshold in range(0,100,0.5):
while threshold<=100:
    print(threshold)

    # In[47]:


    sum_rfaccuracy=0
    count=0
    lowest_accuracy = 100
    whole_dataset = pd.read_csv(filepath+"au.csv", encoding='utf-8',)


    # In[48]:


    # whole_dataset.head()
    # whole_dataset = whole_dataset.drop("Unnamed:0")


    # In[49]:


    # df.loc[:, ~df.columns.str.match('Unnamed')]
    whole_dataset = whole_dataset.loc[:, ~whole_dataset.columns.str.match('Unnamed')]


    # In[50]:


    # whole_dataset.head()
    # print(len(whole_dataset))


    # In[51]:


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
        multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

        return multiple_outliers


    # In[52]:


    # outlier_features = ["popularity","length","danceability","energy","instrumentalness","liveness","loudness","speechiness","valence"]
    # whole_dataset.loc[detect_outliers(whole_dataset,outlier_features)]
    # print(len(whole_dataset.loc[detect_outliers(whole_dataset,outlier_features)]))


    # In[53]:


    # whole_dataset = whole_dataset.drop(detect_outliers(whole_dataset, outlier_features),axis = 0).reset_index(drop = True)


    # In[54]:


    # print(len(whole_dataset))


    # In[55]:


    # threshold = whole_dataset["popularity"].quantile(0.70)

    # df[df.a < df.a.quantile(.95)]

    # threshold_df = whole_dataset.nlargest(int(len(whole_dataset.index)*0.60),'popularity')


    # print(threshold_df.head())

    # print(len(threshold_df))

    # threshold = np.mean(threshold_df.popularity)
    # print(f"type of threshold {type(threshold)}")
    whole_dataset["thresholded_popularity"]= [ 1 if i>=threshold else 0 for i in whole_dataset.popularity ]
    whole_dataset["thresholded_popularity"].value_counts()
    # threshold=80
    # print(f"threshold {threshold}")
    # whole_dataset.loc[whole_dataset['popularity'] < threshold, 'thresholded_popularity'] = 0
    # whole_dataset.loc[whole_dataset['popularity']
    #                   >= threshold, 'thresholded_popularity'] = 1
    # print(whole_dataset.thresholded_popularity.value_counts())


    # In[ ]:





    # In[56]:


    def change_type(var):
        whole_dataset[var] = whole_dataset[var].astype(int)


    # In[57]:


    column= ["key","length","mode"]
    for i in column:
        change_type(i)


    # In[58]:


    x,y = whole_dataset[features] , whole_dataset.loc[:,'thresholded_popularity']
    # x = (x - np.min(x))/(np.max(x)-np.min(x)).values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)
    y=y.astype(int)
    y_train= y_train.astype(int)
    y_test= y_test.astype(int)
    # x_train.info()


    # In[59]:


    whole_dataset.columns[whole_dataset.isnull().any()]

    # song_data.columns[song_data.isnull().any()]
    whole_dataset.isnull().sum()


    # In[64]:


    from sklearn.ensemble import RandomForestClassifier
    rf=RandomForestClassifier(n_estimators=150,random_state = 3)
    rf.fit(x_train,y_train)
    # print("Train ccuracy of random forest",rf.score(x_train,y_train))
    # print("Test accuracy of random forest",rf.score(x_test,y_test))
    acc = rf.score(x_test, y_test)
    print(acc)
    thresholds[threshold]=acc

    RandomForestClassifier_score=rf.score(x_test,y_test)
    y_pred=rf.predict(x_test)
    t_true=y_test


    # In[65]:


    cm = confusion_matrix(y_test, y_pred)
    # print('Confusion matrix: \n', cm)


    # In[86]:


    cr = classification_report(y_test, y_pred,output_dict=True)
    # print('Classification report: \n',cr
            # )


    # In[87]:


    # print(type(cr))
    del(cr["accuracy"])
    sum = 0
    i=0
    for key,value in cr.items():
        if i==2:
            break
        sum+=value["precision"]
        i+=1
    # print(sum)
    precision_dict[threshold] = sum
    if sum>max_sum:
        max_sum=sum
        best_threshold = threshold
    
    threshold+=0.5

print(f"Best threshold:",best_threshold)
print(f"Best Precision:", max_sum)

with open("accuracy_au.json","w",encoding='utf-8') as f:
    json.dump(thresholds,f)

with open("precision_au.json","w",encoding='utf-8') as f:
    json.dump(precision_dict,f)
