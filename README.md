# BS-DNN
Experimental code for BS-DNN
#Data preprocessing：
import numpy as np
import pandas as pd

data = pd.read_csv("Data/churn_train_data.csv", sep="\t")
print(''.join(data['launch_seq'][0].lstrip('[').rstrip(']').split(',')))
print(list(map(int, data['launch_seq'][0].lstrip('[').rstrip(']').split(','))))

temp_array = np.zeros((600001, 32))
for i, item in enumerate(data['launch_seq']):
     temp_array[i] = item.lstrip('[').rstrip(']').split(',')
print(temp_array.dtype)
print(temp_array.shape)
cols = ['launch_seq'+str(i) for i in range(1, 33)]
train = pd.DataFrame(temp_array, columns=cols)
train = pd.concat([data, train], axis=1)
print(train)

temp_array1 = np.zeros((600001, 32))
for i, item in enumerate(data['playtime_seq']):
     temp_array1[i] = item.lstrip('[').rstrip(']').split(',')
print(temp_array1.dtype)
print(temp_array1.shape)
cols = ['playtime_seq'+str(i) for i in range(1, 33)]
train1 = pd.DataFrame(temp_array1, columns=cols)
train = pd.concat([train, train1], axis=1)
print(train)

temp_array2 = np.zeros((600001, 16))
for i, item in enumerate(data['duration_prefer']):
     temp_array2[i] = item.lstrip('[').rstrip(']').split(',')
print(temp_array2.dtype)
print(temp_array2.shape)
cols = ['duration_prefer'+str(i) for i in range(1, 17)]
train2 = pd.DataFrame(temp_array2, columns=cols)
train = pd.concat([train, train2], axis=1)
print(train)

temp_array3 = np.zeros((600001, 11))
for i, item in enumerate(data['interact_prefer']):
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

#feature_engineering：
import random
import numpy as np
import pandas as pd
from itertools import groupby

def launch_grp():
    launch = pd.read_csv('Data/app_launch_logs.csv')
    print(launch.date.min(), launch.date.max())

    launch_grp=launch.groupby("user_id").agg(launch_date=("date",list), launch_type=("launch_type", list)).reset_index()
    print(launch_grp)
    return launch_grp

def choose_end_date(launch_date):
    n1, n2 = min(launch_date), max(launch_date)
    if n1 < n2 - 7:
        end_date = np.random.randint(n1, n2 - 7)
    else:
        end_date = np.random.randint(100, 222 - 7)
    return end_date

def get_label(row):
    launch_list = row.launch_date
    end = row.end_date
    label = sum([1 for x in set(launch_list) if end < x < end + 8])
    if label > 0:
        label = 0
    else:
        label = 1
    return label

def gen_launch_seq(row):
    seq_sort = sorted(zip(row.launch_type, row.launch_date), key=lambda x: x[1])
    seq_map = {k: max(g)[0] + 1 for k, g in groupby(seq_sort, lambda x: x[1])}
    end = row.end_date
    seq = [seq_map.get(x, 0) for x in range(end-31, end+1)]
    return seq

def target_encoding(name, df, m=1):
    df[name] = df[name].str.split(";")
    df = df.explode(name)
    overall = df["label"].mean()
    df=df.groupby(name).agg(freq=("label","count"),in_category=("label", np.mean)).reset_index()
    df["weight"] = df["freq"] / (df["freq"] + m)
    df["score"] = df["weight"] * df["in_category"] + (1 - df["weight"]) * overall
    return df

def get_playtime_seq(row):
    seq_sort = sorted(zip(row.playtime_list, row.date_list), key=lambda x: x[1])
    seq_map = {k: sum(x[0] for x in g) for k, g in groupby(seq_sort, key=lambda x: x[1])}
    seq_norm = {k: 1/(1+np.exp(3-v/450)) for k, v in seq_map.items()}
    seq = [round(seq_norm.get(i, 0), 4) for i in range(row.end_date-31, row.end_date+1)]
    return seq

def get_duration_prefer(duration_list):
    drn_list = sorted(duration_list.split(";"))
    drn_map = {k: sum(1 for _ in g) for k, g in groupby(drn_list) if k != "nan"}
    if drn_map:
        max_ = max(drn_map.values())
        res = [round(drn_map.get(str(i), 0)/max_, 4) for i in range(1, 17)]
        return res
    else:
        return np.nan

def get_id_score(id_list):
    x = sorted(id_list.split(";"))
    x_count = {k: sum(1 for _ in g) for k, g in groupby(x) if k != "nan"}
    if x_count:
        x_sort = sorted(x_count.items(), key=lambda k: -k[1])
        top_x = x_sort[:3]
        res = [(n, id_score.get(k, 0)) for k, n in top_x]
        res = sum(n*v for n, v in res) / sum(n for n, v in res)
        return res
    else:
        return np.nan

def get_interact_prefer(interact_type):
    x = sorted(interact_type)
    x_count = {k: sum(1 for _ in g) for k, g in groupby(x)}
    x_max = max(x_count.values())
    res = [round(x_count.get(i, 0)/x_max, 4) for i in range(1, 12)]
    return res

if __name__ == '__main__':
    launch_grp = launch_grp()
    launch_grp["end_date"] = launch_grp.launch_date.apply(choose_end_date)
    launch_grp["label"] = launch_grp.apply(get_label, axis=1)
    launch_grp.label.value_counts()
    print(launch_grp.label.value_counts())
    print(launch_grp)
    train = launch_grp[["user_id", "end_date", "label"]]
    print(train)

    test = pd.read_csv('Data/test-a.csv')
    test["label"] = -1
    print(test)

    data = pd.concat([train, test], ignore_index=True)
    print(data)

    launch_grp = launch_grp.append(
    test.merge(launch_grp[["user_id", "launch_type", "launch_date"]], how="left", on="user_id"))
    print(launch_grp)

    launch_grp["launch_seq"] = launch_grp.apply(gen_launch_seq, axis=1)
    print(launch_grp)

    data = data.merge(launch_grp[["user_id", "end_date", "label", "launch_seq"]], on=["user_id", "end_date", "label"], how="left")
    print(data)

    playback = pd.read_csv('Data/user_playback_data.csv', dtype={"item_id": str})
    playback = playback.merge(data, how="inner", on="user_id")
    playback = playback.loc[(playback.date >= playback.end_date - 31) & (playback.date <= playback.end_date)]
    print(playback)

    video_data = pd.read_csv('Data/video_related_data.csv', dtype=str)
    playback = playback.merge(video_data[video_data.item_id.notna()], how="left", on="item_id")
    print(playback)

    df = playback.loc[(playback.label >= 0) & (playback.father_id.notna()), ["father_id", "label"]]
    father_id_score = target_encoding("father_id", df)
    print(father_id_score)

    df = playback.loc[(playback.label >= 0) & (playback.tag_list.notna()), ["tag_list", "label"]]
    tag_id_score = target_encoding("tag_list", df)
    tag_id_score.rename({"tag_list": "tag_id"}, axis=1, inplace=True)
    print(tag_id_score)

    df = playback.loc[(playback.label >= 0) & (playback.cast.notna()), ["cast", "label"]]
    cast_id_score = target_encoding("cast", df)
    cast_id_score.rename({"cast": "cast_id"}, axis=1, inplace=True)
    print(cast_id_score)

    playback_grp = playback.groupby(["user_id", "end_date", "label"]).agg(
        playtime_list=("playtime", list),
        date_list=("date", list),
        duration_list=("duration", lambda x: ";".join(map(str, x))),
        father_id_list=("father_id", lambda x: ";".join(map(str, x))),
        tag_list=("tag_list", lambda x: ";".join(map(str, x))),
        cast_list=("cast", lambda x: ";".join(map(str, x)))
    ).reset_index()
    print(playback_grp)

    playback_grp["playtime_seq"] = playback_grp.apply(get_playtime_seq, axis=1)
    print(playback_grp)

    drn_desc = video_data.loc[video_data.duration.notna(), "duration"].astype(int)
    print(drn_desc.min(), drn_desc.max())

    playback_grp["duration_prefer"] = playback_grp.duration_list.apply(get_duration_prefer)
    print(playback_grp)

    id_score = dict()
    id_score.update({x[1]: x[5] for x in father_id_score.itertuples()})
    id_score.update({x[1]: x[5] for x in tag_id_score.itertuples()})
    id_score.update({x[1]: x[5] for x in cast_id_score.itertuples()})

    father_id_score.shape[0] + tag_id_score.shape[0] + cast_id_score.shape[0] == len(id_score)
    playback_grp["father_id_score"] = playback_grp.father_id_list.apply(get_id_score)
    playback_grp["cast_id_score"] = playback_grp.cast_list.apply(get_id_score)
    playback_grp["tag_score"] = playback_grp.tag_list.apply(get_id_score)
    print(playback_grp)

    data = data.merge(
        playback_grp[
            ["user_id", "end_date", "label", "playtime_seq", "duration_prefer", "father_id_score", "cast_id_score",
             "tag_score"]], on=["user_id", "end_date", "label"], how="left")
    print(data)

    portrait = pd.read_csv("Data/user_portrait_data.csv", dtype={"territory_code": str})
    portrait = pd.merge(data[["user_id", "label"]], portrait, how="left", on="user_id")
    print(portrait)

    df = portrait.loc[(portrait.label >= 0) & (portrait.territory_code.notna()), ["territory_code", "label"]]
    territory_score = target_encoding("territory_code", df)
    print(territory_score)
    n1 = len(id_score)
    id_score.update({x[1]: x[5] for x in territory_score.itertuples()})
    n1 + territory_score.shape[0] == len(id_score)
    portrait["territory_score"] = portrait.territory_code.apply(lambda x: id_score.get(x, 0) if isinstance(x, str) else np.nan)
    print(portrait)

    portrait["device_ram"] = portrait.device_ram.apply(lambda x: float(x.split(";")[0]) if isinstance(x, str) else np.nan)
    portrait["device_rom"] = portrait.device_rom.apply(lambda x: float(x.split(";")[0]) if isinstance(x, str) else np.nan)
    print(portrait)
    data = data.merge(portrait.drop("territory_code", axis=1), how="left", on=["user_id", "label"])
    print(data)

    interact = pd.read_csv("Data/user_interaction_data.csv")
    print(interact.interact_type.min(), interact.interact_type.max())
    interact_grp = interact.groupby("user_id").agg(interact_type=("interact_type", list)).reset_index()
    print(interact_grp)

    interact_grp["interact_prefer"] = interact_grp.interact_type.apply(get_interact_prefer)
    print(interact_grp)
    data = data.merge(interact_grp[["user_id", "interact_prefer"]], on="user_id", how="left")
    print(data)

    norm_cols = ["father_id_score", "cast_id_score", "tag_score",
                 "device_type", "device_ram", "device_rom", "sex",
                 "age", "education", "occupation_status", "territory_score"]
    for col in norm_cols:
        mean = data[col].mean()
        std = data[col].std()
        data[col] = (data[col] - mean) / std
    print(data)

    data.fillna({"playtime_seq": str([0] * 32), "duration_prefer": str([0] * 16), "interact_prefer": str([0] * 11)}, inplace=True)
    print(data)
    data.fillna(0, inplace=True)
    print(data)

    data.loc[data.label >= 0].to_csv("Data/churn_train_data.csv", sep="\t", index=False)
data.loc[data.label < 0].to_csv("Data/churn_test_data.csv", sep="\t", index=False)
