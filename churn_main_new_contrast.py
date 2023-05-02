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
#LR:
#          F1       AUC       ACC    Recall  Precision
# 0  0.731158  0.755022  0.693055  0.830702   0.652919
# 1  0.730589  0.754162  0.693967  0.827693   0.653876
# 2  0.732519  0.751556  0.692833  0.829613   0.655770
# 3  0.729574  0.751845  0.691300  0.828800   0.651567
# 4  0.732862  0.754668  0.693967  0.835307   0.652800
# 5  0.733942  0.752005  0.693767  0.834228   0.655180
# 6  0.732770  0.756465  0.693750  0.831974   0.654704
# 7  0.732668  0.754652  0.692617  0.833625   0.653522
# 8  0.730786  0.752808  0.691167  0.834273   0.650140
# 9  0.733087  0.752688  0.692850  0.835303   0.653160
# F1           0.731995
# AUC          0.753587
# ACC          0.692927
# Recall       0.832152
# Precision    0.653364

#RF:
#          F1       AUC       ACC    Recall  Precision
# 0  0.712698  0.763357  0.694455  0.754246   0.675489
# 1  0.723348  0.772158  0.701350  0.778790   0.675276
# 2  0.721238  0.768623  0.697250  0.772511   0.676347
# 3  0.721460  0.769829  0.699850  0.773668   0.675853
# 4  0.722850  0.772162  0.700517  0.777137   0.675653
# 5  0.722838  0.771642  0.698900  0.775470   0.676896
# 6  0.724924  0.774081  0.702533  0.776659   0.679652
# 7  0.725100  0.774488  0.701733  0.778507   0.678550
# 8  0.723537  0.772794  0.701250  0.778080   0.676141
# 9  0.708552  0.766831  0.693450  0.737936   0.681418
# F1           0.720655
# AUC          0.770597
# ACC          0.699129
# Recall       0.770300
# Precision    0.677127

# KNN
#         F1       AUC       ACC    Recall  Precision
# 0  0.002845  0.500043  0.497592  0.001426   0.518072
# 1  0.496397  0.495650  0.495650  0.495811   0.496984
# 2  0.022723  0.500765  0.493933  0.011605   0.542243
# 3  0.656819  0.499461  0.501650  0.949181   0.502150
# 4  0.499052  0.502363  0.502317  0.493284   0.504957
# 5  0.499711  0.495729  0.495750  0.497383   0.502060
# 6  0.505639  0.503198  0.503200  0.503418   0.507879
# 7  0.501613  0.497928  0.497950  0.500016   0.503220
# 8  0.498627  0.497821  0.497817  0.497015   0.500250
# 9  0.670115  0.499665  0.504600  0.996435   0.504799
# F1           0.435354
# AUC          0.499262
# ACC          0.499046
# Recall       0.494557
# Precision    0.508261

# Bayes
#          F1       AUC       ACC    Recall  Precision
# 0  0.700824  0.647734  0.576374  0.987495   0.543147
# 1  0.699239  0.643193  0.574533  0.986536   0.541534
# 2  0.705964  0.637788  0.584650  0.983497   0.550593
# 3  0.700959  0.635140  0.578417  0.983414   0.544553
# 4  0.701244  0.639346  0.577683  0.986237   0.544034
# 5  0.703876  0.637864  0.580150  0.985516   0.547431
# 6  0.702482  0.636597  0.579117  0.984545   0.546046
# 7  0.701933  0.640107  0.576467  0.986971   0.544640
# 8  0.698403  0.641647  0.570883  0.988887   0.539829
# 9  0.701855  0.655305  0.575233  0.990098   0.543600
# F1           0.701678
# AUC          0.641472
# ACC          0.577351
# Recall       0.986320
# Precision    0.544541
