import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error, r2_score, f1_score, log_loss, roc_auc_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier

from keras.activations import relu, tanh, sigmoid
from keras.optimizers import adam, rmsprop, sgd, nadam
from keras.losses import poisson
from modeller_utilities import neural_network, null_scaler, poisson_loss, poisson_loss_xgb, TimeDistributedScaler, \
    lstm, cnn1d
from sklearn.model_selection import KFold, GroupKFold, GridSearchCV, RandomizedSearchCV
from hyperopt import hp, tpe, anneal


LOGGER_PATH = os.environ['SCTYS_PROJECT'] + '/Log/log_sctys_modeller'
NOTIFIER_PATH = os.environ['SCTYS_PROJECT'] + '/sctys_notify'
TEMP_PATH = os.environ['SCTYS_PROJECT'] + '/tmp'
IO_PATH = os.environ['SCTYS_PROJECT'] + '/sctys_io'
VISUAL_PATH = os.environ['SCTYS_PROJECT'] + '/sctys_visualization'
NOTIFIER_AGENT = 'slack'


class ModellerSetting(object):
    LOGGER_FILE = 'modeller.log'
    LOGGER_LEVEL = 'DEBUG'
    MODEL_PATH = os.environ['SCTYS_DATA'] + '/sctys_modeller'
    SCALER_DICT = {'standard': StandardScaler, 'minmax': MinMaxScaler, 'maxabs': MaxAbsScaler, 'null': null_scaler,
                   'timed': TimeDistributedScaler}
    ESTIMATOR_DICT = {'linear': LinearRegression, 'ridge': Ridge, 'lasso': Lasso, 'logistic': LogisticRegression,
                      'rf_reg': RandomForestRegressor, 'rf_clf': RandomForestClassifier, 'xgb_reg': XGBRegressor,
                      'xgb_clf': XGBClassifier, 'lgbm_reg': LGBMRegressor, 'lgbm_clf': LGBMClassifier,
                      'cat_reg': CatBoostRegressor, 'cat_clf': CatBoostClassifier, 'nn_reg': neural_network,
                      'nn_clf': neural_network, 'lstm_reg': lstm, 'cnn_reg': cnn1d, 'huber': HuberRegressor}
    SEARCHER_DICT = {'grid': GridSearchCV, 'random': RandomizedSearchCV, 'tpe': tpe.suggest, 'anneal': anneal.suggest}
    SCORER_DICT = {'mse': {'func': mean_squared_error, 'greater_better': False},
                   'r2': {'func': r2_score, 'greater_better': True}, 'f1': {'func': f1_score, 'greater_better': True},
                   'logloss': {'func': log_loss, 'greater_better': False},
                   'roc_auc': {'func': roc_auc_score, 'greater_better': True},
                   'poisson': {'func': poisson_loss, 'greater_better': False},
                   'poisson_xgb': {'func': poisson_loss_xgb, 'greater_better': False}}
    CV_DICT = {'kfold': KFold, 'groupkfold': GroupKFold}
    HYPEROPT_MAX_ITER = 20
    HYPEROPT_SPACE = {
        'ridge': {
            'estimator__alpha': {
                'dist': hp.loguniform('estimator__alpha', -10, 10), 'dtype': np.float32}},
        'lasso': {
            'estimator__alpha': {
                'dist': hp.loguniform('estimator__alpha', -10, 10), 'dtyoe': np.float32}},
        'logistic': {
            'estimator__C': {'dist': hp.loguniform('estimator__C', -10, 10), 'dtype': np.float32}},
        'rf_reg': {
            'estimator__n_estimators': {'dist': hp.quniform('estimator__n_estimators', 100, 500, 50),
                                        'dtype': np.uint16},
            'estimator__max_depth': {'dist': hp.quniform('estimator__max_depth', 10, 30, 1), 'dtype': np.uint16},
            'estimator__min_samples_split': {'dist': hp.quniform('estimator__min_samples_split', 2, 10, 1),
                                             'dtype': np.uint16},
            'estimator__min_samples_leaf': {'dist': hp.quniform('estimator__min_samples_leaf', 2, 10, 1),
                                            'dtype': np.uint16}},
        'xgb_reg': {
            'estimator__max_depth': {'dist': hp.quniform('estimator__max_depth', 3, 8, 1), 'dtype': np.uint16},
            'estimator__learning_rate': {'dist': hp.loguniform('estimator__learning_rate', -10, -1),
                                         'dtype': np.float32},
            'estimator__n_estimators': {'dist': hp.quniform('estimator__n_estimators', 100, 500, 50),
                                        'dtype': np.uint16},
            'estimator__gamma': {'dist': hp.quniform('estimator__gamma', 0.5, 1, 0.05), 'dtype': np.float32},
            'estimator__min_child_weight': {'dist': hp.quniform('estimator__min_child_weight', 1, 20, 1),
                                            'dtype': np.uint16},
            'estimator__max_delta_step': {'dist': hp.quniform('estimator__max_delta_step', 1, 20, 1),
                                          'dtype': np.uint16},
            'estimator__colsample_bytree': {'dist': hp.quniform('estimator__colsample_bytree', 0.5, 1, 0.05),
                                            'dtype': np.float32},
            'estimator__colsample_bylevel': {'dist': hp.quniform('estimator__colsample_bylevel', 0.5, 1, 0.05),
                                             'dtype': np.float32},
            'estimator__colsample_bynode': {'dist': hp.quniform('estimator__colsample_bynode', 0.5, 1, 0.05),
                                            'dtype': np.float32}},
        'nn_reg': {
            'estimator__hidden_no_unit':
                {'dist': hp.quniform('estimator__hidden_no_unit', 5, 50, 5), 'dtype': np.int32},
            'estimator__hidden_layer': {'dist': hp.quniform('estimator__hidden_layer', 1, 5, 1), 'dtype': np.int32},
            'estimator__activation': {'dist': hp.choice('estimator__activation', (relu, tanh, sigmoid)),
                                      'dtype': None, 'choice': (relu, tanh, sigmoid)},
            'estimator__optimizer': {'dist': hp.choice('estimator__optimizer', (adam, rmsprop, sgd, nadam)),
                                     'dtype': None, 'choice': (adam, rmsprop, sgd, nadam)},
            'estimator__lr': {'dist': hp.loguniform('estimator__lr', -10, -1), 'dtype': np.float32},
            'estimator__drop_out_rate': {'dist': hp.uniform('estimator__drop_out_rate', 0.1, 0.5), 'dtype': np.float32},
            # 'estimator__batch_size': {'dist': hp.choice('estimator__batch_size', (32, 64, 128)), 'dtype': np.uint16},
            # 'estimator__epochs': {'dist': hp.quniform('estimator__epochs', 3, 10, 1), 'dtype': np.uint8}
        },
        'lstm_reg': {
            'estimator__lstm_no_unit':
                {'dist': hp.quniform('estimator__lstm_no_unit', 5, 50, 5), 'dtype': np.int32},
            'estimator__lstm_layer': {'dist': hp.quniform('estimator__lstm_layer', 1, 3, 1), 'dtype': np.int32},
            'estimator__dense_no_unit':
                {'dist': hp.quniform('estimator__dense_no_unit', 5, 50, 5), 'dtype': np.int32},
            'estimator__dense_layer': {'dist': hp.quniform('estimator__dense_layer', 1, 3, 1), 'dtype': np.int32},
            'estimator__lstm_activation': {'dist': hp.choice('estimator__lstm_activation', (relu, tanh, sigmoid)),
                                           'dtype': None, 'choice': (relu, tanh, sigmoid)},
            'estimator__dense_activation': {'dist': hp.choice('estimator__dense_activation', (relu, tanh, sigmoid)),
                                            'dtype': None, 'choice': (relu, tanh, sigmoid)},
            'estimator__optimizer': {'dist': hp.choice('estimator__optimizer', (adam, rmsprop, sgd, nadam)),
                                     'dtype': None, 'choice': (adam, rmsprop, sgd, nadam)},
            'estimator__lr': {'dist': hp.loguniform('estimator__lr', -10, -1), 'dtype': np.float32},
            'estimator__drop_out_rate': {'dist': hp.uniform('estimator__drop_out_rate', 0.1, 0.5), 'dtype': np.float32},
        },
        'cnn_reg': {
            'estimator__cnn_no_filter':
                {'dist': hp.quniform('estimator__cnn_no_filter', 5, 50, 5), 'dtype': np.int32},
            'estimator__cnn_layer': {'dist': hp.quniform('estimator__cnn_layer', 1, 3, 1), 'dtype': np.int32},
            'estimator__cnn_kernal': {'dist': hp.quniform('estimator__cnn_kernal', 1, 2, 1), 'dtype': np.int32},
            'estimator__cnn_pooling': {'dist': hp.quniform('estimator__cnn_pooling', 1, 2, 1), 'dtype': np.int32},
            'estimator__dense_no_unit':
                {'dist': hp.quniform('estimator__dense_no_unit', 5, 50, 5), 'dtype': np.int32},
            'estimator__dense_layer': {'dist': hp.quniform('estimator__dense_layer', 1, 3, 1), 'dtype': np.int32},
            'estimator__cnn_activation': {'dist': hp.choice('estimator__cnn_activation', (relu, tanh, sigmoid)),
                                          'dtype': None, 'choice': (relu, tanh, sigmoid)},
            'estimator__dense_activation': {'dist': hp.choice('estimator__dense_activation', (relu, tanh, sigmoid)),
                                            'dtype': None, 'choice': (relu, tanh, sigmoid)},
            'estimator__optimizer': {'dist': hp.choice('estimator__optimizer', (adam, rmsprop, sgd, nadam)),
                                     'dtype': None, 'choice': (adam, rmsprop, sgd, nadam)},
            'estimator__lr': {'dist': hp.loguniform('estimator__lr', -10, -1), 'dtype': np.float32},
            'estimator__drop_out_rate': {'dist': hp.uniform('estimator__drop_out_rate', 0.1, 0.5), 'dtype': np.float32},
        }
    }



