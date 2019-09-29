import os
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.layers import Input, Dense, BatchNormalization, Activation, Dropout, LSTM, Masking, Conv1D, MaxPooling1D, \
    Flatten
from keras.models import Model
from keras.utils.vis_utils import plot_model


def set_logger(logger_path, logger_file_name, logger_level, logger_name):
    logger_file = os.path.join(logger_path, logger_file_name)
    logging.basicConfig(level=getattr(logging, logger_level),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(filename=logger_file), logging.StreamHandler()])
    logger = logging.getLogger(logger_name)
    return logger


def null_scaler():
    scaler = StandardScaler(with_mean=False, with_std=False)
    return scaler


def get_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


class TimeDistributedScaler(object):

    def __init__(self, scaler_name, mask_value):
        scaler_dict = {'standard': StandardScaler, 'minmax': MinMaxScaler, 'maxabs': MaxAbsScaler, 'null': null_scaler}
        self.scaler = scaler_dict[scaler_name]()
        self.mask_value = mask_value
        self.shape = ()

    @ staticmethod
    def get_last_time_tensor(x_tensor):
        x_matrix = x_tensor[:, -1, :]
        return x_matrix

    def from_tensor_to_matrix(self, x_tensor):
        self.shape = x_tensor.shape
        x_matrix = x_tensor.reshape(-1, self.shape[-1])
        return x_matrix

    def from_matrix_to_tensor(self, x_matrix):
        x_tensor = x_matrix.reshape(self.shape)
        return x_tensor

    def fit(self, x_tensor, y=None):
        x_matrix = self.get_last_time_tensor(x_tensor)
        # x_matrix = np.unique(x_matrix, axis=0)
        x_matrix = x_matrix.astype(np.float32)
        x_matrix[x_matrix == self.mask_value] = np.nan
        self.scaler = self.scaler.fit(x_matrix)
        return self.scaler

    def transform(self, x, y=None):
        if len(x.shape) > 2:
            x_matrix = self.from_tensor_to_matrix(x)
        else:
            x_matrix = x
        x_scaled = self.scaler.transform(x_matrix)
        x_scaled[np.isnan(x_scaled)] = self.mask_value
        x_scaled = self.from_matrix_to_tensor(x_scaled)
        return x_scaled


def poisson_loss(y_true, y_pred):
    log_loss = -np.nanmean(-y_pred + y_true * np.log(y_pred + 10**-8))
    return log_loss


def poisson_loss_xgb(predt, dtrain):
    y = dtrain
    predt = np.where(predt < 0.001, 0.001, predt)
    grad = 1 - y/predt
    hess = y/(predt**2)
    return grad, hess


def poisson_exp_loss(y_true, y_pred):
    log_loss = -np.sum(-np.exp(y_pred) + y_true * y_pred)
    return log_loss


def neural_network(input_dim, output_dim, hidden_no_unit, hidden_layer, activation, out_activation, loss, optimizer,
                   metric, drop_out_rate, use_normalization=True, model_graphic_full_file_name=None):
    # ktf.clear_session()
    ktf.set_session(get_session())
    inputs = Input(shape=(input_dim,))
    nodes = None
    hidden_list = [hidden_no_unit] * hidden_layer
    activation_list = [activation] * len(hidden_list)
    for layer, act_func in zip(hidden_list, activation_list):
        if nodes is None:
            nodes = Dense(layer)(inputs)
        else:
            nodes = Dense(layer)(nodes)
        if use_normalization:
            nodes = BatchNormalization()(nodes)
        nodes = Activation(act_func)(nodes)
        nodes = Dropout(drop_out_rate)(nodes)
    nodes = Dense(output_dim, activation=out_activation)(nodes)
    model = Model(inputs=inputs, outputs=nodes)
    print(model.summary())
    if model_graphic_full_file_name is not None:
        plot_model(model, to_file=model_graphic_full_file_name, show_shapes=True, show_layer_names=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metric)
    return model


def lstm(input_dim, output_dim, lstm_no_unit, lstm_layer, dense_no_unit, dense_layer, lstm_activation, dense_activation,
         out_activation, loss, optimizer, metric, drop_out_rate, max_time, mask_value, use_normalization=True,
         model_graphic_full_file_name=None):
    # K.clear_session()
    ktf.set_session(get_session())
    inputs = Input(shape=(max_time, input_dim))
    nodes = Masking(mask_value=mask_value)(inputs)
    lstm_list = [lstm_no_unit] * lstm_layer
    dense_list = [dense_no_unit] * dense_layer
    lstm_activation_list = [lstm_activation] * lstm_layer
    dense_activation_list = [dense_activation] * dense_layer
    for layer, act_func in zip(lstm_list[:-1], lstm_activation_list[:-1]):
        nodes = LSTM(layer, return_sequences=True, unroll=False)(nodes)
        if use_normalization:
            nodes = BatchNormalization()(nodes)
        nodes = Activation(act_func)(nodes)
        nodes = Dropout(drop_out_rate)(nodes)
    nodes = LSTM(lstm_list[-1], unroll=False)(nodes)
    if use_normalization:
        nodes = BatchNormalization()(nodes)
    nodes = Activation(lstm_activation_list[-1])(nodes)
    nodes = Dropout(drop_out_rate)(nodes)
    for layer, act_func in zip(dense_list, dense_activation_list):
        nodes = Dense(layer)(nodes)
        if use_normalization:
            nodes = BatchNormalization()(nodes)
        nodes = Activation(act_func)(nodes)
        nodes = Dropout(drop_out_rate)(nodes)
    nodes = Dense(output_dim, activation=out_activation)(nodes)
    model = Model(inputs=inputs, outputs=nodes)
    print(model.summary())
    if model_graphic_full_file_name is not None:
        plot_model(model, to_file=model_graphic_full_file_name, show_shapes=True, show_layer_names=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metric)
    return model


def cnn1d(input_dim, output_dim, cnn_no_filter, cnn_layer, cnn_kernal, cnn_pooling, dense_no_unit, dense_layer,
          cnn_activation, dense_activation, out_activation, loss, optimizer, metric, drop_out_rate, max_time,
          mask_value, use_normalization=True, model_graphic_full_file_name=None):
    # ktf.clear_session()
    ktf.set_session(get_session())
    inputs = Input(shape=(max_time, input_dim))
    # nodes = Masking(mask_value=mask_value)(inputs)
    nodes = None
    cnn_filter_list = [cnn_no_filter] * cnn_layer
    cnn_kernal_list = [(cnn_kernal,)] * cnn_layer
    cnn_pooling_list = [(cnn_pooling,)] * cnn_layer
    dense_list = [dense_no_unit] * dense_layer
    cnn_activation_list = [cnn_activation] * cnn_layer
    dense_activation_list = [dense_activation] * dense_layer
    for layer, kernal, pool, act_func in zip(
            cnn_filter_list[:-1], cnn_kernal_list[:-1], cnn_pooling_list[:-1], cnn_activation_list[:-1]):
        if nodes is None:
            nodes = Conv1D(layer, kernal, padding='same')(inputs)
        else:
            nodes = Conv1D(layer, kernal, padding='same')(nodes)
        if use_normalization:
            nodes = BatchNormalization()(nodes)
        nodes = Activation(act_func)(nodes)
        nodes = MaxPooling1D(pool, padding='same')(nodes)
        nodes = Dropout(drop_out_rate)(nodes)
    if nodes is None:
        nodes = Conv1D(cnn_filter_list[-1], cnn_kernal_list[-1], padding='same')(inputs)
    else:
        nodes = Conv1D(cnn_filter_list[-1], cnn_kernal_list[-1], padding='same')(nodes)
    if use_normalization:
        nodes = BatchNormalization()(nodes)
    nodes = Activation(cnn_activation_list[-1])(nodes)
    nodes = MaxPooling1D(cnn_pooling_list[-1])(nodes)
    nodes = Dropout(drop_out_rate)(nodes)
    nodes = Flatten()(nodes)
    for layer, act_func in zip(dense_list, dense_activation_list):
        nodes = Dense(layer)(nodes)
        if use_normalization:
            nodes = BatchNormalization()(nodes)
        nodes = Activation(act_func)(nodes)
        nodes = Dropout(drop_out_rate)(nodes)
    nodes = Dense(output_dim, activation=out_activation)(nodes)
    model = Model(inputs=inputs, outputs=nodes)
    print(model.summary())
    if model_graphic_full_file_name is not None:
        plot_model(model, to_file=model_graphic_full_file_name, show_shapes=True, show_layer_names=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metric)
    return model


