import argparse

from inputs import SparseFeat, DenseFeat
from deepfm import *
import numpy as np
import gc
import keras
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from tqdm import tqdm
from sklearn.metrics import f1_score
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))

parser = argparse.ArgumentParser(description='nn')
parser.add_argument('--l', type=int, default=14)
parser.add_argument('--bs', type=int, default=1024)
parser.add_argument('--att', action='store_true')

args = parser.parse_args()


def multi_category_focal_loss2(gamma=2., alpha=.25):
    """
    focal loss for multi category of multi label problem
    适用于多分类或多标签问题的focal loss
    alpha控制真值y_true为1/0时的权重
        1的权重为alpha, 0的权重为1-alpha
    当你的模型欠拟合，学习存在困难时，可以尝试适用本函数作为loss
    当模型过于激进(无论何时总是倾向于预测出1),尝试将alpha调小
    当模型过于惰性(无论何时总是倾向于预测出0,或是某一个固定的常数,说明没有学到有效特征)
        尝试将alpha调大,鼓励模型进行预测出1。
    Usage:
     model.compile(loss=[multi_category_focal_loss2(alpha=0.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)

    def multi_category_focal_loss2_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
        ce = -tf.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
        loss = tf.reduce_mean(fl)
        return loss

    return multi_category_focal_loss2_fixed


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


class Metrics(Callback):

    def on_epoch_end(self, epoch, logs={}):
        # 获取验证集F1
        val_targ = label_validate
        val_pre = self.model.predict(val_model_input, batch_size=2048)
        threshold_test = np.percentile(val_pre, 89.4)
        val_pre = [1 if i > threshold_test else 0 for i in val_pre]
        f1 = f1_score(val_targ, val_pre)
        print(f'val f1: {f1}')

        # 对test做预测，并根据阈值进行调整，最终输出每个epoch的提交文件，最后根据验证集F1结果选择合适的预测文件提交到线上
        test['target'] = self.model.predict(test_model_input, verbose=0, batch_size=2048)
        sub = test[['id', 'target']]
        threshold_test = np.percentile(sub['target'], 89.4)
        sub['target'] = [1 if i > threshold_test else 0 for i in sub['target']]
        sub.to_csv('sub_l{}_{:02d}_bs{}_att{}_{:.5f}.csv'.format(args.l, epoch + 1, args.bs, args.att, f1), index=False)

        return


train = pd.read_csv("../data/train.csv")
train['isTest'] = 0
test = pd.read_csv("../data/test.csv")
test['isTest'] = 1
data = train.append(test).reset_index(drop=True)
del train, test
gc.collect()

gc.collect()

data = data.sort_values(['deviceid', 'ts']).reset_index(drop=True)
data['device_vendor'] = data['device_vendor'].apply(lambda x: x.strip().lower())

data.drop(['timestamp', 'lng', 'lat', 'osversion', 'guid'], axis=1, inplace=True)
# lstm部分输入特征（还有gap）
cate_feats = ['pos', 'newsid']
cate2nunique = {}
for cate in cate_feats:
    data[cate] = LabelEncoder().fit_transform(data[cate])
    cate2nunique[cate] = data[cate].nunique() + 1
# deepfm输入特征
cate_feats_concat = ['netmodel', 'device_vendor', 'device_version', 'app_version'] + ['deviceid', 'newsid', 'pos']
print(cate_feats_concat)
cate_concat2nunique = {}
for cate in cate_feats_concat:
    data[cate] = LabelEncoder().fit_transform(data[cate])
    cate_concat2nunique[cate] = data[cate].nunique() + 1

data = reduce_mem_usage(data)

group = data.groupby('deviceid')['ts']
data['gap'] = group.shift(0) - group.shift(1)
del group
gc.collect()

data['gap'] = data['gap'].fillna(data['gap'].mean())
data['gap'] = np.log(data['gap'] + 1)
sclaer = StandardScaler()
data[['gap']] = np.float16(sclaer.fit_transform(data[['gap']]))

# 当前记录前后窗口长度（14）
l = args.l
timing_cols = []  # 序列特征列表
len_pos = data['pos'].nunique() + 1
print(f"pos:{data['pos'].unique()}")

# 构造序列特征
for i in tqdm(range(l * 2 + 1)):
    data['gap_%s' % (i - l)] = data['gap'].shift(i - l)
    data['gap_%s' % (i - l)] = data['gap_%s' % (i - l)].fillna(0)
    timing_cols += ['gap_%s' % (i - l)]

    for cate in cate_feats:
        new_col = f'{cate}_{(i - l)}'
        if cate in ['pos', 'newsid']:
            data[new_col] = data[[cate]].shift(i - l).fillna(cate2nunique[cate] - 1)
        else:
            data[new_col] = data[cate]
        timing_cols += [new_col]

data[timing_cols] = reduce_mem_usage(data[timing_cols])
data = data.sort_values(['ts']).reset_index(drop=True)

train = data[data['isTest'] != 1]
test = data[data['isTest'] == 1]
del data
train_data_use = np.array(train[timing_cols]).reshape(len(train), l * 2 + 1, len(cate_feats) + 1)  # lstm input
train_label = train['target'].values
train_data_sideinfo = train[cate_feats_concat].values  # deepfm input
del train
test_data_use = np.array(test[timing_cols]).reshape(len(test), l * 2 + 1, len(cate_feats) + 1)  # lstm input
test_data_sideinfo = test[cate_feats_concat].values  # deepfm input
# del test

test = test[['id']]
gc.collect()

# 训练集验证集按照 4:1比例分割
train_size = int(len(train_data_use) * 0.8)
# 训练集、验证集 lstm特征
X_train, X_validate = train_data_use[:train_size], train_data_use[train_size:]
label_train, label_validate = train_label[:train_size], train_label[train_size:]

# 训练集、验证集 deepfm的特征
X_train_side, X_validate_side = train_data_sideinfo[:train_size], train_data_sideinfo[train_size:]

sparse_features = cate_feats_concat
dense_features = []

# 构造模型，可参考deepctr deepfm例子
fixlen_feature_columns = [SparseFeat(feat, cate_concat2nunique[feat])
                          for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', att=args.att, seq_len=l * 2 + 1,
               cate_feats=cate_feats,
               cate2nunique=cate2nunique)
model.compile(loss=multi_category_focal_loss2(alpha=0.106535), optimizer='adam', metrics=['acc'])
print(model.summary())

plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.1, patience=5)
early_stopping = EarlyStopping(monitor='val_acc', patience=9, mode='max')
train_model_input = {'lstm_input': X_train}
val_model_input = {'lstm_input': X_validate}
test_model_input = {'lstm_input': test_data_use}
for i, cate in enumerate(cate_feats_concat):
    train_model_input[cate] = X_train_side[:, i]
    val_model_input[cate] = X_validate_side[:, i]
    test_model_input[cate] = test_data_sideinfo[:, i]
# 开始训练，Metrics会保存了每个epoch的预测结果
history = model.fit(train_model_input, label_train, epochs=50, batch_size=args.bs,
                    verbose=2, shuffle=True,
                    validation_data=(val_model_input, label_validate),
                    callbacks=[early_stopping, plateau, Metrics()])
