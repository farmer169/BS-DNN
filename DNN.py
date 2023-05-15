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
                # seq = [row.launch_seq, row.playtime_seq]
                # fea = list(row[7:18])
                # input_1.append(np.array(seq))
                # input_2.append(np.array(fea))
                feature = list(row[2:])
                input_2.append(np.array(feature))
                output.append(row.label)
                if len(input_2) == self.batch_size or idx == self.num - 1:
                # #   input_1 = np.array(input_1).transpose([0, 2, 1])
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
    data = data[['label', 'father_id_score', 'cast_id_score', 'tag_score', 'device_type',
                 'device_ram', 'device_rom', 'sex', 'age', 'education', 'occupation_status', 'territory_score',
                 'duration_prefer1', 'duration_prefer2', 'duration_prefer3', 'duration_prefer4', 'duration_prefer5',
                 'duration_prefer6', 'duration_prefer7', 'duration_prefer8', 'duration_prefer9', 'duration_prefer10',
                 'duration_prefer11', 'duration_prefer12', 'duration_prefer13', 'duration_prefer14', 'duration_prefer15',
                 'duration_prefer16', 'interact_prefer1', 'interact_prefer2', 'interact_prefer3', 'interact_prefer4',
                 'interact_prefer5', 'interact_prefer6', 'interact_prefer7', 'interact_prefer8', 'interact_prefer9',
                 'interact_prefer10', 'interact_prefer11']]
    # from sklearn.preprocessing import StandardScaler
    # label = data['label']
    # data = data.drop(columns=['label'])
    # ss = StandardScaler()
    # for col in data.columns:
    #     data[col] = ss.fit_transform(data[col].values.reshape(-1, 1))
    # data['label'] = label
    # print(data)

    # data["launch_seq"] = data.launch_seq.apply(lambda x: json.loads(x))
    # data["playtime_seq"] = data.playtime_seq.apply(lambda x: json.loads(x))
    # data["duration_prefer"] = data.duration_prefer.apply(lambda x: json.loads(x))
    # data["interact_prefer"] = data.interact_prefer.apply(lambda x: json.loads(x))

    kf = KFold(n_splits=10, random_state=12, shuffle=False)
    result_dict = {'F1': [], 'AUC': [], 'ACC': [], 'Recall': [], 'Precision': []}
    i = 0
    for train_index, test_index in kf.split(data):
        i += 1
        print("-------------------{}------------------".format(i))
        train, test_data = data.loc[train_index], data.loc[test_index]
        # train, test_data = data[train_index, :], data[test_index, :]
        train, dev = train_test_split(train, test_size=0.05, random_state=42)
        print(train.shape, type(train))
        test_data = test_data.reset_index(drop=True)
        # print(test_data)
        test = DataGenerator(test_data, 64)
        dev = DataGenerator(dev, 64)
        train = DataGenerator(train, 256)

        model = build_model(feature_num=38)

        # scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        #     boundaries=[2, 5, 8],
        #     values=[0.01, 0.008, 0.005, 0.003]
        # )
        # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=scheduler), loss="binary_crossentropy",
        #               metrics=['accuracy', keras.metrics.AUC()])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy",
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

    # prediction = model.predict(iter(dev), steps=len(dev))
    # print(prediction)
    # predictions = [1 if prediction[i][0] > 0.5 else 0 for i in range(prediction.shape[0])]
    # print(predictions)
    # print(predictions.count(1))
    # print(predictions.count(0))
    # y_test = data.loc[:30000]['label']
    #
    # print(y_test)
    # auc = roc_auc_score(y_test, prediction)
    # precision = precision_score(y_test, predictions)
    # recall = recall_score(y_test, predictions)
    # f1 = f1_score(y_test, predictions)
    # print('F1: %f  Precision:%f  Recall:%f  AUC:%f' % (f1, precision, recall, auc))
    # F1: 0.673999  Precision:0.636058  Recall:0.716754  AUC:0.652044


    #              0.0005              0.001                 0.01               0.005             0.1
    # F1           0.656822           0.687562              0.375739            0.598199         0.433741
    # AUC          0.681290           0.697995              0.543554            0.603801         0.500468
    # ACC          0.644161           0.658517              0.544814            0.596872         0.504694
    # Recall       0.695767           0.746024              0.428221            0.626944         0.573681
    # Precision    0.631356           0.637759              0.424844            0.594903         0.355614



