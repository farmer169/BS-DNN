
import numpy as np
import pandas as pd

data = pd.read_csv("Data/churn_train_data.csv", sep="\t")
print(''.join(data['launch_seq'][0].lstrip('[').rstrip(']').split(',')))
print(list(map(int, data['launch_seq'][0].lstrip('[').rstrip(']').split(','))))

temp_array = np.zeros((600001, 32))
for i, item in enumerate(data['launch_seq']):
    # temp_array[i] = list(map(int, item.lstrip('[').rstrip(']').split(',')))
    temp_array[i] = item.lstrip('[').rstrip(']').split(',')
print(temp_array.dtype)
print(temp_array.shape)
cols = ['launch_seq'+str(i) for i in range(1, 33)]
train = pd.DataFrame(temp_array, columns=cols)
train = pd.concat([data, train], axis=1)
print(train)

temp_array1 = np.zeros((600001, 32))
for i, item in enumerate(data['playtime_seq']):
    # temp_array[i] = list(map(int, item.lstrip('[').rstrip(']').split(',')))
    temp_array1[i] = item.lstrip('[').rstrip(']').split(',')
print(temp_array1.dtype)
print(temp_array1.shape)
cols = ['playtime_seq'+str(i) for i in range(1, 33)]
train1 = pd.DataFrame(temp_array1, columns=cols)
train = pd.concat([train, train1], axis=1)
print(train)

temp_array2 = np.zeros((600001, 16))
for i, item in enumerate(data['duration_prefer']):
    # temp_array[i] = list(map(int, item.lstrip('[').rstrip(']').split(',')))
    temp_array2[i] = item.lstrip('[').rstrip(']').split(',')
print(temp_array2.dtype)
print(temp_array2.shape)
cols = ['duration_prefer'+str(i) for i in range(1, 17)]
train2 = pd.DataFrame(temp_array2, columns=cols)
train = pd.concat([train, train2], axis=1)
print(train)

temp_array3 = np.zeros((600001, 11))
for i, item in enumerate(data['interact_prefer']):
    # temp_array[i] = list(map(int, item.lstrip('[').rstrip(']').split(',')))
    temp_array3[i] = item.lstrip('[').rstrip(']').split(',')
print(temp_array3.dtype)
print(temp_array3.shape)
cols = ['interact_prefer'+str(i) for i in range(1, 12)]
train3 = pd.DataFrame(temp_array3, columns=cols)
train = pd.concat([train, train3], axis=1)
print(train)
train = train.drop(columns=['launch_seq', 'playtime_seq', 'duration_prefer', 'interact_prefer'])
print(train)
train.to_csv('Data/churn_data_nosqe.csv', index=False)