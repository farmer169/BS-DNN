import keras.metrics
import pandas as pd
import numpy as np
import json
import math
import tensorflow as tf
from tensorflow.keras import backend as k
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
np.random.seed(12)
tf.random.set_seed(12)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


if __name__ == '__main__':
    data = pd.read_csv("Data/churn_data_nosqe.csv")

    # clf = DecisionTreeClassifier(random_state=12, max_depth=9)
    # clf = LogisticRegression(random_state=12, penalty='l1', solver='liblinear')
    # clf = RandomForestClassifier(random_state=2, n_jobs=-1, n_estimators=100, verbose=1)
    # clf = KNeighborsClassifier(n_neighbors=1)
    clf = GaussianNB()
    kf = KFold(n_splits=10, random_state=12, shuffle=False)
    result_dict = {'F1': [], 'AUC': [], 'ACC': [], 'Recall': [], 'Precision': []}
    i = 0
    for train_index, test_index in kf.split(data):
        i = i + 1
        print("-------------------{}------------------".format(i))
        train, test_data = data.loc[train_index], data.loc[test_index]
        y_train = train['label']
        x_train = train.drop(columns=['label'])
        y_test = test_data['label']
        x_test = test_data.drop(columns=['label'])
        clf.fit(x_train, y_train)
        proba = clf.predict_proba(x_test)
        pred = clf.predict(x_test)

        auc = roc_auc_score(y_test, proba[:, 1])
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        acc = accuracy_score(y_test, pred)

        result_dict['F1'].append(f1)
        result_dict['AUC'].append(auc)
        result_dict['ACC'].append(acc)
        result_dict['Recall'].append(recall)
        result_dict['Precision'].append(precision)

        print('F1: %f  AUC:%f  ACC:%f  Recall:%f  Precision:%f' % (f1, auc, acc, recall, precision))
    df_result = pd.DataFrame(result_dict)
    print(df_result)
    df_result.to_csv('Data/{}_results.csv'.format('GaussianNB'), index=False)
    print(df_result.mean().T)
