# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)

"""

import tensorflow as tf
from inputs import build_input_features, input_from_feature_columns, get_linear_logit, combined_dnn_input
from deepctr.layers import FM, DNN, PredictionLayer
from deepctr.layers.utils import concat_fun
from tensorflow.python.keras import Input
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Lambda, Concatenate, Embedding, LSTM, Permute, Dense, multiply


def attention_3d_block(inputs, seq_len=21):
    # input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(seq_len, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


def DeepFM(linear_feature_columns, dnn_feature_columns, embedding_size=8, use_fm=True, dnn_hidden_units=(128, 128),
           l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0,
           dnn_activation='relu', dnn_use_bn=False, task='binary', att=False, seq_len=None, cate_feats=[],
           cate2nunique={}):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param embedding_size: positive integer,sparse feature embedding_size
    :param use_fm: bool,use FM part or not
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    features = build_input_features(linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list, embedding_dict = input_from_feature_columns(features, dnn_feature_columns,
                                                                                         embedding_size,
                                                                                         l2_reg_embedding, init_std,
                                                                                         seed)

    linear_logit = get_linear_logit(features, linear_feature_columns, l2_reg=l2_reg_linear, init_std=init_std,
                                    seed=seed, prefix='linear')

    fm_input = concat_fun(sparse_embedding_list, axis=1)
    fm_logit = FM()(fm_input)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

    input_lstm = Input(shape=(seq_len, 1+len(cate_feats)), name='lstm_input')
    input_lstm_gap = Lambda(lambda x: x[:, :, 0:1])(input_lstm)
    concate_list = [input_lstm_gap]
    for i, cate in enumerate(cate_feats):
        input_cate = Lambda(lambda x: x[:, :, i + 1])(input_lstm)
        emb = embedding_dict.get(cate)
        if emb is None:
            emb = Embedding(output_dim=8, input_dim=cate2nunique[cate])
        concate_list.append(emb(input_cate))
    input_lstm_concat = Concatenate(axis=-1)(concate_list)
    if att:
        lstm_out = LSTM(units=128, return_sequences=True)(input_lstm_concat)
        attention_mul = attention_3d_block(lstm_out, seq_len)
        lstm_out = Lambda(lambda x: K.sum(x, axis=1))(attention_mul)
    else:
        lstm_out = LSTM(units=128, return_sequences=False)(input_lstm_concat)

    dnn_input = concat_fun([dnn_input, lstm_out])
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  dnn_use_bn, seed)(dnn_input)
    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(dnn_out)

    if len(dnn_hidden_units) == 0 and use_fm == False:  # only linear
        final_logit = linear_logit
    elif len(dnn_hidden_units) == 0 and use_fm == True:  # linear + FM
        final_logit = tf.keras.layers.add([linear_logit, fm_logit])
    elif len(dnn_hidden_units) > 0 and use_fm == False:  # linear +　Deep
        final_logit = tf.keras.layers.add([linear_logit, dnn_logit])
    elif len(dnn_hidden_units) > 0 and use_fm == True:  # linear + FM + Deep
        final_logit = tf.keras.layers.add([linear_logit, fm_logit, dnn_logit])
    else:
        raise NotImplementedError

    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list + [input_lstm], outputs=output)
    return model
