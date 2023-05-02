import keras.metrics
import pandas as pd
import numpy as np
import json
import math
import tensorflow as tf
from tensorflow.keras import backend as k
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from keras import metrics
from sklearn.model_selection import KFold, train_test_split
np.random.seed(24)
tf.random.set_seed(24)

class DataGenerator:
    def __init__(self, df, batch_size):
        self.data = df
        self.num = df.shape[0]
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(self.num / self.batch_size)

    def __iter__(self):
        while True:
            input_2, output = [], []
            for row in self.data.itertuples():
                idx = row.Index
                feature = list(row[:104])
                input_2.append(np.array(feature))
                output.append(row.label)
                if len(input_2) == self.batch_size or idx == self.num - 1:
                    input_2 = np.array(input_2)
                    output = np.array(output)
                    yield (input_2), output
                    input_2, output = [], []

def build_model(feature_num):
    # input_1 = tf.keras.Input(shape=(seq_len, seq_feature_num))
    # output_1 = tf.keras.layers.GRU(64)(input_1)

    input_2 = tf.keras.Input(shape=(feature_num, ))
    layer = tf.keras.layers.Dense(256, activation="elu")(input_2)
    layer = tf.keras.layers.Dense(128, activation="elu")(layer)
    output = tf.keras.layers.Dense(64, activation="elu")(layer)

    # output = tf.concat([output_1, output_2], -1)
    output = tf.keras.layers.Dense(1, activation="relu")(output)

    model = tf.keras.Model(inputs=[input_2], outputs=output)
    return model

if __name__ == '__main__':
    data = pd.read_csv("Data/churn_data_nosqe.csv")
    from sklearn.preprocessing import StandardScaler
    label = data['label']
    data = data.drop(columns=['label'])
    ss = StandardScaler()
    for col in data.columns:
        data[col] = ss.fit_transform(data[col].values.reshape(-1, 1))
    data['label'] = label
    print(data)

    kf = KFold(n_splits=10, random_state=12, shuffle=False)
    result_dict = {'F1': [], 'AUC': [], 'ACC': [], 'Recall': [], 'Precision': []}
    i = 0
    for train_index, test_index in kf.split(data):
        i += 1
        print("-------------------{}------------------".format(i))
        train, test_data = data.loc[train_index], data.loc[test_index]
        train, dev = train_test_split(train, test_size=0.05, random_state=42)
        print(train.shape, type(train))
        test_data = test_data.reset_index(drop=True)
        # print(test_data)
        test = DataGenerator(test_data, 64)
        dev = DataGenerator(dev, 64)
        train = DataGenerator(train, 256)

        model = build_model(feature_num=104)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss="binary_crossentropy",
                      metrics=['accuracy', keras.metrics.AUC()])
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        model.fit(iter(train), steps_per_epoch=len(train), validation_data=iter(dev), validation_steps=len(dev),
                  epochs=20, callbacks=[early_stopping])
        # callbacks=[early_stopping]
        model.save("Data/churn_mlp_best_weights.h5")
        prediction = model.predict(iter(test), steps=len(test))
        predictions = [1 if prediction[i][0] > 0.5 else 0 for i in range(prediction.shape[0])]
        print(predictions.count(1), predictions.count(0))
        print()

        y_test = test_data['label'].values

        auc = roc_auc_score(y_test, prediction)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        acc = accuracy_score(y_test, predictions)

        result_dict['F1'].append(f1)
        result_dict['AUC'].append(auc)
        result_dict['ACC'].append(acc)
        result_dict['Recall'].append(recall)
        result_dict['Precision'].append(precision)

        print('F1: %f  AUC:%f  ACC:%f  Recall:%f  Precision:%f' % (f1, auc, acc, recall, precision))
        print(pd.DataFrame(result_dict))

    df_result = pd.DataFrame(result_dict)
    print(df_result)
    df_result.to_csv('Data/dnn_results.csv', index=False)
    print(df_result.mean().T)




