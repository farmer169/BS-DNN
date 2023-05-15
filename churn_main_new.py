import keras.metrics
import pandas as pd
import numpy as np
import json
import math
import tensorflow as tf
from tensorflow.keras import backend as k
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import KFold, train_test_split
np.random.seed(12)
tf.random.set_seed(12)


class DataGenerator:
    def __init__(self, df, batch_size):
        self.data = df
        self.num = df.shape[0]
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(self.num / self.batch_size)

    def __iter__(self):
        while True:
            input_1, input_2, output = [], [], []
            for row in self.data.itertuples():
                idx = row.Index
                seq = [row.launch_seq, row.playtime_seq]
                fea = row.duration_prefer + row.interact_prefer + list(row[7:18])
                input_1.append(np.array(seq))
                input_2.append(np.array(fea))
                output.append(row.label)
                if len(input_1) == self.batch_size or idx == self.num - 1:
                    input_1 = np.array(input_1).transpose([0, 2, 1])
                    input_2 = np.array(input_2)
                    output = np.array(output)
                    yield (input_1, input_2), output
                    input_1, input_2, output = [], [], []

# def build_model(seq_feature_num, seq_len, feature_num):
#     input_1 = tf.keras.Input(shape=(seq_len, seq_feature_num))
#     output_1 = tf.keras.layers.GRU(64)(input_1)
#
#     input_2 = tf.keras.Input(shape=(feature_num, ))
#
#     output = tf.concat([output_1, input_2], -1)
#     layer1 = tf.keras.layers.Dense(256, activation="elu")(output)
#     layer2 = tf.keras.layers.Dense(128, activation="elu")(layer1)
#     layer3 = tf.keras.layers.Dense(64, activation="elu")(layer2)
#     output = tf.keras.layers.Dense(1, activation="relu")(layer3)
#
#     model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)
#     return model


def build_model(seq_feature_num, seq_len, feature_num) -> object:
    input_1 = tf.keras.Input(shape=(seq_len, seq_feature_num))
    output_1 = tf.keras.layers.Conv1D(8, kernel_size=6, strides=1)(input_1)
    output_1 = tf.keras.layers.GRU(64)(output_1)

    input_2 = tf.keras.Input(shape=(feature_num, ))
    layer = tf.keras.layers.Dense(256, activation="elu")(input_2)
    layer = tf.keras.layers.Dense(128, activation="elu")(layer)
    output_2 = tf.keras.layers.Dense(64, activation="elu")(layer)

    output = tf.concat([output_1, output_2], -1)
    output = tf.keras.layers.Dense(64, activation="elu")(output)
    output = tf.keras.layers.Dense(32, activation="elu")(output)
    output = tf.keras.layers.Dense(1, activation="relu")(output)

    model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)
    return model


if __name__ == '__main__':
    data = pd.read_csv("Data/churn_train_data.csv", sep="\t")
    data["launch_seq"] = data.launch_seq.apply(lambda x: json.loads(x))
    data["playtime_seq"] = data.playtime_seq.apply(lambda x: json.loads(x))
    data["duration_prefer"] = data.duration_prefer.apply(lambda x: json.loads(x))
    data["interact_prefer"] = data.interact_prefer.apply(lambda x: json.loads(x))
    print(data['label'].value_counts())
    # for row in data[:2].itertuples():
    #     idx = row.Index
    #     seq = [row.launch_seq, row.playtime_seq]
    #     fea = row.duration_prefer + row.interact_prefer + list(row[7:18])
    #     print(row[1])
    #     print(list(row[7:18]))

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
        #print(test_data)
        test = DataGenerator(test_data, 64)
        dev = DataGenerator(dev, 64)
        train = DataGenerator(train, 128)
        model = build_model(seq_feature_num=2, seq_len=32, feature_num=38)

        scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[2, 5, 8],
            values=[0.001, 0.0008, 0.0005, 0.0003]
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=scheduler), loss="binary_crossentropy",
                      metrics=['accuracy', keras.metrics.AUC()])

        # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008), loss="binary_crossentropy",
        #               metrics=['accuracy', keras.metrics.AUC()])
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        model.fit(iter(train), steps_per_epoch=len(train), validation_data=iter(dev), validation_steps=len(dev),
                  epochs=20, callbacks=[early_stopping])
        # callbacks=[early_stopping]
        model.save("Data/churn_best_weights.h5")

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
    df_result.to_csv('Data/deep_results.csv', index=False)
    print(df_result.mean())
    #F1: 0.736674  Precision:0.672559  Recall:0.814303  AUC:0.778378  ACC:0.708033 (64,1, patience=1)
    #F1: 0.739863  Precision:0.668905  Recall:0.827662  AUC:0.782224  ACC:0.708100 (64,32,1, patience=1)
    #F1: 0.735499  Precision:0.675969  Recall:0.806527  AUC:0.782450  ACC:0.709067 (64,32,1, patience=2)
    #F1: 0.735338  Precision:0.658228  Recall:0.832912  AUC:0.758696  ACC:0.699300 (32,1, patience=1)
    #F1: 0.734523  Precision:0.670378  Recall:0.812242  AUC:0.768061  ACC:0.705533 (64,32,16, 1, patience=1)
    #F1: 0.685339  Precision:0.523277  Recall:0.992822  AUC:0.552452  ACC:0.542767 (128, 64,32,1, patience=1)
    #F1: 0.176371  Precision:0.577434  Recall:0.104081  AUC:0.514966  ACC:0.512467 (256, 128, 64,32,1, patience=1)
    #F1: 0.727800  Precision:0.650757  Recall:0.825535  AUC:0.749153  ACC:0.690300 (128,64,1, patience=1)



    #epochs=2:F1: 0.710853  Precision:0.650979  Recall:0.782857  AUC:0.674523
    #epochs=3:F1: 0.743403  Precision:0.644077  Recall:0.878949  AUC:0.690336
    #epochs=4:F1: 0.735479  Precision:0.665255  Recall:0.822278  AUC:0.701749
    #epochs=5(128固定):F1: 0.724195  Precision:0.655484  Recall:0.808998  AUC:0.690577
    #                 F1: 0.731880  Precision:0.666831  Recall:0.810992  AUC:0.701638
    #                 F1: 0.735316  Precision:0.663812  Recall:0.824085  AUC:0.702065
    #                 F1: 0.736652  Precision:0.662858  Recall:0.828936  AUC:0.702351
    #                 F1: 0.728370  Precision:0.667294  Recall:0.801755  AUC:0.699760
    #                 F1: 0.734840  Precision:0.667935  Recall:0.816641  AUC:0.704061
    #                 F1: 0.705619  Precision:0.662736  Recall:0.754436  AUC:0.684059
    #                 F1: 0.726799  Precision:0.674033  Recall:0.788529  AUC:0.702410
    #                 F1: 0.707320  Precision:0.683636  Recall:0.732704  AUC:0.695762
    #                 F1: 0.656239  Precision:0.534810  Recall:0.849006  AUC:0.552964
    #                 F1: 0.734398  Precision:0.659679  Recall:0.828205  AUC:0.699143
    #                 F1: 0.726351  Precision:0.675409  Recall:0.785605  AUC:0.771266
    #epochs=5(128): F1: 0.750707  Precision:0.632301  Recall:0.923678  AUC:0.685107
    #               F1: 0.723279  Precision:0.675456  Recall:0.778391  AUC:0.701207
    #epochs=5(128,42):F1: 0.730266  Precision:0.670431  Recall:0.801827  AUC:0.701122
    #                 F1: 0.704422  Precision:0.613788  Recall:0.826458  AUC:0.649637
    #                 F1: 0.734844  Precision:0.662152  Recall:0.825465  AUC:0.699248
    #                 F1: 0.544203  Precision:0.649435  Recall:0.468318  AUC:0.606021
    #                 F1: 0.631857  Precision:0.646614  Recall:0.617758  AUC:0.637749
    #                 F1: 0.726842  Precision:0.633297  Recall:0.852811  AUC:0.676103
    #                 F1: 0.733438  Precision:0.653053  Recall:0.836390  AUC:0.692963
    #                 F1: 0.662940  Precision:0.565492  Recall:0.800967  AUC:0.588529
    #                 F1: 0.731185  Precision:0.661908  Recall:0.816659  AUC:0.696892
    #                 F1: 0.714986  Precision:0.655233  Recall:0.786731  AUC:0.683539
    #                 F1: 0.741134  Precision:0.651539  Recall:0.859299  AUC:0.696699
    #epochs=5(128,1):F1: 0.638638  Precision:0.502582  Recall:0.875703  AUC:0.497714
    #epochs=5(128,2):F1: 0.742636  Precision:0.665319  Recall:0.840286  AUC:0.701032
    #                F1: 0.742879  Precision:0.671194  Recall:0.831707  AUC:0.704652
    #                F1: 0.746228  Precision:0.663894  Recall:0.851876  AUC:0.702381
    #                F1: 0.737447  Precision:0.670399  Recall:0.819396  AUC:0.700873
    #                F1: 0.737378  Precision:0.669498  Recall:0.820575  AUC:0.700308
    #                F1: 0.746879  Precision:0.668356  Recall:0.846310  AUC:0.705470
    #                F1: 0.739349  Precision:0.665705  Recall:0.831314  AUC:0.699262
    #                F1: 0.737902  Precision:0.675294  Recall:0.813306  AUC:0.703938
    #                F1: 0.745887  Precision:0.664110  Recall:0.850632  AUC:0.702302
    #                F1: 0.741185  Precision:0.665088  Recall:0.836946  AUC:0.700007
    #                F1: 0.741971  Precision:0.658595  Recall:0.849519  AUC:0.696484
    #                F1: 0.733960  Precision:0.675834  Recall:0.803025  AUC:0.764310
    #                F1: 0.736811  Precision:0.673963  Recall:0.812586  AUC:0.763518
    #epochs=5(128,3):F1: 0.726729  Precision:0.671512  Recall:0.791841  AUC:0.700937
    #epochs=5(128,4):F1: 0.734148  Precision:0.672319  Recall:0.808499  AUC:0.701885
    #epochs=5(128,5):F1: 0.726101  Precision:0.660082  Recall:0.806792  AUC:0.693533
    #epochs=5(128,6):F1: 0.737847  Precision:0.665508  Recall:0.827830  AUC:0.698437
    #epochs=5(128,7):F1: 0.745910  Precision:0.618059  Recall:0.940452  AUC:0.678496
    #                F1: 0.743560  Precision:0.653043  Recall:0.863207  AUC:0.701392
    #                F1: 0.732384  Precision:0.671099  Recall:0.805988  AUC:0.704711
    #epochs=5(128,8):F1: 0.602047  Precision:0.611547  Recall:0.592837  AUC:0.604753
    #epochs=5(128,9):F1: 0.726020  Precision:0.675645  Recall:0.784511  AUC:0.694198
    #      （10万）：F1: 0.730261  Precision:0.654448  Recall:0.825939  AUC:0.693419
    #               F1: 0.719715  Precision:0.662201  Recall:0.788169  AUC:0.691672
    #epochs=5(256固定):F1: 0.668043  Precision:0.501550  Recall:1.000000  AUC:0.500000
    #                 F1: 0.715641  Precision:0.644515  Recall:0.804413  AUC:0.678989
    #                 F1: 0.743798  Precision:0.646926  Recall:0.874792  AUC:0.697193
    #                 F1: 0.733072  Precision:0.657795  Recall:0.827806  AUC:0.697239
    #epochs=5（256）：F1: 0.736214  Precision:0.659707  Recall:0.832793  AUC:0.698712
    #epochs=5（512）：F1: 0.731159  Precision:0.590536  Recall:0.959685  AUC:0.646843;
    #                F1: 0.719400  Precision:0.621166  Recall:0.854541  AUC:0.657442
    #epochs=10:Precision:0.642981  Recall:0.829495  AUC:0.678374; F1: 0.736780  Precision:0.665415  Recall:0.825291  AUC:0.704479
    #epochs=10（固定）:F1: 0.676872  Precision:0.624319  Recall:0.739084  AUC:0.645789
    #epochs=10（256固定）：F1: 0.733393  Precision:0.668490  Recall:0.812255  AUC:0.703473
    #                    F1: 0.729928  Precision:0.674386  Recall:0.795441  AUC:0.704495
    #epochs=15（256固定）F1: 0.728661  Precision:0.681012  Recall:0.783478  AUC:0.707106
    #                   F1: 0.739441  Precision:0.658567  Recall:0.842959  AUC:0.701605
    #epochs=20:Precision:0.635987  Recall:0.813481  AUC:0.670736
    # data = pd.read_csv('Data/test_data.csv', sep='\t')
    # data['launch_seq'] = data.launch_seq.apply(lambda x: json.loads(x))
    # data['playtime_seq'] = data.playtime_seq.apply(lambda x: json.loads(x))
    # data['duration_prefer'] = data.duration_prefer.apply(lambda x: json.loads(x))
    # data['interact_prefer'] = data.interact_prefer.apply(lambda x: json.loads(x))
    # test = DataGenerator(data, 64)
    # prediction = model.predict(iter(test), steps=len(test))
    # print(prediction)
    #
    # data['prediction'] = np.reshape(prediction, -1)
    # data = data[['user_id', 'prediction']]
    # print(data)
    # data.to_csv('Data/submission.csv', index=False, header=False, float_format='%.2f')