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
    # data = data.drop(columns=['user_id', 'end_date'])
    data = data[['label','father_id_score','cast_id_score','tag_score','device_type',
                 'device_ram','device_rom','sex','age','education','occupation_status','territory_score','duration_prefer1',
                 'duration_prefer2','duration_prefer3','duration_prefer4','duration_prefer5','duration_prefer6',
                 'duration_prefer7','duration_prefer8','duration_prefer9','duration_prefer10','duration_prefer11',
                 'duration_prefer12','duration_prefer13','duration_prefer14','duration_prefer15','duration_prefer16',
                 'interact_prefer1','interact_prefer2','interact_prefer3','interact_prefer4','interact_prefer5',
                 'interact_prefer6','interact_prefer7','interact_prefer8','interact_prefer9','interact_prefer10','interact_prefer11']]

    # clf = DecisionTreeClassifier(random_state=12, max_depth=9)
    # clf = LogisticRegression(random_state=12, penalty='l1', solver='liblinear')
    # clf = RandomForestClassifier(random_state=2, n_jobs=-1, n_estimators=200, verbose=1, max_depth=9)
    clf = KNeighborsClassifier(n_neighbors=1)
    # clf = GaussianNB()
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
    df_result.to_csv('Data/{}_results.csv'.format('knn'), index=False)
    # df_result.to_csv('Data/{}_{}_results.csv'.format('seq','knn'), index=False)
    print(df_result.mean().T)

    #            tree            seq_tree          tree+27
    # F1         0.686623        0.730109          0.701100
    # AUC        0.702716        0.746152          0.711047
    # ACC        0.656907        0.693764          0.661531
    # Recall     0.745875        0.821971          0.787757
    # Precision  0.636121        0.656731          0.631652


#LR:                             seq_lr            lr+27
# F1           0.610186          0.732308          0.684411
# AUC          0.633150          0.753945          0.698283
# ACC          0.601576          0.693194          0.659719
# Recall       0.618786          0.832752          0.732192
# Precision    0.601830          0.653489          0.642489

#RF:                                                                                                           seq_RF      RF+27
#             max_depth=5     max_depth=4      max_depth=6     max_depth=7      max_depth=8    max_depth=9
# F1           0.682817       0.681478         0.683379        0.684012         0.684273       0.684636        0.740462    0.695260
# AUC          0.699317       0.696838         0.702107        0.704440         0.706620       0.708363        0.758750    0.713840
# ACC          0.654806       0.651427         0.656789        0.658014         0.658779       0.659269        0.697559    0.663936
# Recall       0.737312       0.739939         0.734977        0.734498         0.733745       0.733928        0.856116    0.760733
# Precision    0.635830       0.631583         0.638557        0.640026         0.641057       0.641555        0.652339    0.640169

# KNN                         seq_knn          knn+27
# F1           0.574538       0.629457         0.579407
# AUC          0.568271       0.610967         0.572874
# ACC          0.568351       0.611316         0.572961
# Recall       0.578334       0.655119         0.583696
# Precision    0.570795       0.605736         0.575188

# Bayes                         seq_Bayes      Bayes+27
# F1           0.618153         0.721874       0.674965
# AUC          0.627865         0.699641       0.676435
# ACC          0.606589         0.660432       0.531127
# Recall       0.631897         0.874456       0.966049
# Precision    0.605008         0.614632       0.518685
