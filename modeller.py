import sys
import pandas as pd
import numpy as np
from modeller_setting import LOGGER_PATH, NOTIFIER_PATH, TEMP_PATH, IO_PATH, VISUAL_PATH, NOTIFIER_AGENT, \
    ModellerSetting
from modeller_utilities import set_logger, get_session
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_validate, validation_curve, learning_curve
from sklearn.metrics import make_scorer
from hyperopt import fmin, tpe, hp, anneal, Trials
import keras.backend as K
import keras.backend.tensorflow_backend as ktf
import tensorflow as tf
sys.path.append(NOTIFIER_PATH)
sys.path.append(IO_PATH)
sys.path.append(VISUAL_PATH)
from notifiers import send_message
from file_io import FileIO
from visualization import Visualization


class Modeller(ModellerSetting):

    def __init__(self):
        self.file_io = FileIO()
        self.visual = Visualization()
        self.logger = set_logger(LOGGER_PATH, self.LOGGER_FILE, self.LOGGER_LEVEL, __name__)
        ktf.set_session(get_session())

    def save_estimator(self, estimator, file_name):
        self.file_io.save_file(estimator, self.MODEL_PATH, file_name, 'joblib')

    def load_estimator(self, file_name):
        estimator = self.file_io.load_file(self.MODEL_PATH, file_name, 'joblib')
        return estimator

    @ staticmethod
    def notify_model_fitting_failure(message):
        send_message('Error in fitting model. {}'.format(message), NOTIFIER_AGENT)

    def try_save_notify_exit(self, func, estimator, x_data, y_data, *args, model_file_name=None, verb=False,
                             **kwargs):
        try:
            estimator = func(estimator, x_data, y_data, *args, **kwargs)
            if model_file_name is not None:
                self.save_estimator(estimator, model_file_name)
            if verb:
                self.logger.debug('Model {} fitting completed'.format(model_file_name))
            return estimator
        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:
            self.notify_model_fitting_failure('{}. model_file_name: {}'.format(e, model_file_name))
            sys.exit()

    def set_feature_scaler(self, name, **kwargs):
        if name in self.SCALER_DICT:
            scaler = self.SCALER_DICT[name](**kwargs)
        else:
            self.logger.error('Unknown scaler name {}'.format(name))
            scaler = None
        return scaler

    @ staticmethod
    def feature_scaling(scaler, x_data, fit=True):
        if fit:
            model = scaler.fit(x_data)
            return model
        else:
            transformed_data = scaler.transform(x_data)
            return transformed_data

    def set_estimator(self, name, **kwargs):
        if name in self.ESTIMATOR_DICT:
            estimator = self.ESTIMATOR_DICT[name](**kwargs)
        else:
            self.logger.error('Unknown estimator name {}'.format(name))
            estimator = None
        return estimator

    @ staticmethod
    def _estimator_fit(estimator, x_data, y_data, **kwargs):
        estimator = estimator.fit(x_data, y_data, **kwargs)
        return estimator

    def pure_estimation(self, estimator, x_data, y_data, fit=True, model_file_name=None, verb=False, **kwargs):
        if fit:
            estimator = self.try_save_notify_exit(self._estimator_fit, estimator, x_data, y_data,
                                                  model_file_name=model_file_name, verb=verb, **kwargs)
            return estimator
        else:
            # ktf.get_session().run(tf.global_variables_initializer())
            estimated_data = estimator.predict(x_data)
            return estimated_data

    def model_residual(self, estimator, x_data, y_data, fit=False, model_file_name=None, verb=False, **kwargs):
        if fit:
            estimator = self.pure_estimation(estimator, x_data, y_data, True, model_file_name, verb, **kwargs)
        estimated_data = self.pure_estimation(estimator, x_data, y_data, False, model_file_name, verb)
        residual_data = y_data - estimated_data
        return residual_data

    @staticmethod
    def metric_to_scorer(metric, **kwargs):
        return make_scorer(metric, **kwargs)

    def set_scorer(self, name, make_score=True, **kwargs):

        if name in self.SCORER_DICT:
            scorer = self.SCORER_DICT[name]
            if make_score:
                scorer = self.metric_to_scorer(scorer['func'], greater_is_better=scorer['greater_better'], **kwargs)
            else:
                scorer = scorer['func']
        else:
            self.logger.error('Unknown scorer name {}'.format(name))
            scorer = None
        return scorer

    def model_scoring(self, estimator, x_data, y_data, metric, fit=False, model_file_name=None, verb=False, **kwargs):
        if fit:
            estimator = self.pure_estimation(estimator, x_data, y_data, True, model_file_name, verb, **kwargs)
        estimated_data = self.pure_estimation(estimator, x_data, y_data, False, model_file_name, verb)
        score = metric(y_data, estimated_data)
        return score

    @ staticmethod
    def set_estimation_pipeline(scaler, estimator):
        estimator = Pipeline([('scaler', scaler), ('estimator', estimator)])
        return estimator

    def train_valid_evaluation(self, estimator, x_train, y_train, x_valid, y_valid, scorer, fit=False,
                               model_file_name=None, verb=False, **kwargs):
        if fit:
            estimator = self.pure_estimation(estimator, x_train, y_train, True, model_file_name, verb, **kwargs)
        train_score = self.model_scoring(estimator, x_train, y_train, scorer, False)
        valid_score = self.model_scoring(estimator, x_valid, y_valid, scorer, False)
        score = {'train': train_score, 'valid': valid_score}
        return score

    def set_cv(self, cv_name, **kwargs):
        if cv_name in self.CV_DICT:
            cv = self.CV_DICT[cv_name](**kwargs)
        else:
            self.logger.error('Unknown scorer name {}'.format(cv_name))
            cv = None
        return cv

    @ staticmethod
    def _cross_validation(estimator, x_data, y_data, scorer, **kwargs):
        estimator = cross_validate(estimator, x_data, y_data, scoring=scorer, **kwargs)
        return estimator

    def cross_validation(self, estimator, x_data, y_data, scorer, model_file_name=None, verb=False, **kwargs):
        estimator = self.try_save_notify_exit(self._cross_validation, estimator, x_data, y_data, scorer,
                                              model_file_name=model_file_name, verb=verb, **kwargs)
        return estimator

    def validation_curve(self, estimator, x_data, y_data, para_range_dict, scorer, plot_file_name, **kwargs):
        para_name = list(para_range_dict.keys())[0]
        para_values = para_range_dict[para_name]
        train_score, valid_score = validation_curve(estimator, x_data, y_data, para_name, para_values, scoring=scorer,
                                                    **kwargs)
        train_score = np.mean(train_score, axis=1)
        valid_score = np.mean(valid_score, axis=1)
        data = pd.DataFrame({**para_range_dict, **{'train_score': train_score, 'valid_score': valid_score}})
        self.visual.time_series_plot(plot_file_name, data, para_name, ['train_score', 'valid_score'],
                                     title_dict={'title': 'Validation curve for parameter {}'.format(para_name),
                                                 'x_title': para_name, 'y_title': 'score'})
        return data

    def learning_curve(self, estimator, x_data, y_data, scorer, plot_file_name, **kwargs):
        train_sizes, train_score, valid_score = learning_curve(estimator, x_data, y_data, scoring=scorer, **kwargs)
        data = pd.DataFrame({'train_size': train_sizes, 'train_score': train_score, 'valid_score': valid_score})
        self.visual.time_series_plot(plot_file_name, data, 'train_size', ['train_score', 'valid_score'],
                                     title_dict={'title': 'Learning curve', 'x_title': 'train_size',
                                                 'y_title': 'score'})
        return data

    def para_search(self, estimator, x_data, y_data, method, search_para, scorer, model_file_name=None, verb=False,
                    **kwargs):
        searcher = self.SEARCHER_DICT[method]
        estimator = searcher(estimator, search_para, scoring=scorer, **kwargs)
        estimator = self.pure_estimation(estimator, x_data, y_data, True, model_file_name, verb)
        return estimator

    def hyperopt_search(self, scaler_dict, estimator_dict, x_data, y_data, method, search_para, scorer,
                        model_file_name=None, verb=False, **kwargs):

        set_feature_scaler = self.set_feature_scaler
        set_estimator = self.set_estimator
        set_estimation_pipeline = self.set_estimation_pipeline

        def hyperopt_min_func(space):
            scaler_kwargs = scaler_dict['kwargs']
            estimator_kwargs = estimator_dict['kwargs']
            for param_key, param_value in space.items():
                if 'scaler' in param_key:
                    if param_value['dtype'] is None:
                        scaler_kwargs[param_key.replace('scaler__', '')] = param_value['dist']
                    else:
                        scaler_kwargs[param_key.replace('scaler__', '')] = param_value['dtype'](param_value['dist'])
                if 'estimator' in param_key:
                    if param_value['dtype'] is None:
                        estimator_kwargs[param_key.replace('estimator__', '')] = param_value['dist']
                    else:
                        estimator_kwargs[param_key.replace('estimator__', '')] = param_value['dtype'](
                            param_value['dist'])
            if 'optimizer' in estimator_kwargs and 'lr' in estimator_kwargs:
                estimator_kwargs['optimizer'] = estimator_kwargs['optimizer'](lr=estimator_kwargs['lr'])
                estimator_kwargs.pop('lr')
            scaler = set_feature_scaler(scaler_dict['name'], **scaler_dict['kwargs'])
            estimator = set_estimator(estimator_dict['name'], **estimator_dict['kwargs'])
            model = set_estimation_pipeline(scaler, estimator)
            score = -cross_val_score(model, x_data, y_data, scoring=scorer, **kwargs).mean()
            if 'nn' in estimator_dict['name'] or 'lstm' in estimator_dict['name'] or 'cnn' in estimator_dict['name']:
                K.clear_session()
            return score

        searcher = self.SEARCHER_DICT[method]
        trials = Trials()
        best = fmin(fn=hyperopt_min_func, space=search_para, algo=searcher, max_evals=self.HYPEROPT_MAX_ITER,
                    trials=trials)
        log = trials.trials
        self.save_estimator(log, 'tmp_hyperpt_search_log.pkl')
        best_para = {key: best[key] for key in search_para.keys()}
        self.save_estimator(best_para, 'temp_hyperopt_best_para.pkl')
        scaler_kwargs = scaler_dict['kwargs']
        estimator_kwargs = estimator_dict['kwargs']
        for param_key, param_value in best_para.items():
            if 'scaler' in param_key:
                if search_para[param_key]['dtype'] is None:
                    scaler_kwargs[param_key.replace('scaler__', '')] = search_para[param_key]['choice'][param_value]
                else:
                    scaler_kwargs[param_key.replace('scaler__', '')] = search_para[param_key]['dtype'](param_value)
            if 'estimator' in param_key:
                if search_para[param_key]['dtype'] is None:
                    estimator_kwargs[param_key.replace('estimator__', '')] = search_para[param_key]['choice'][
                        param_value]
                else:
                    estimator_kwargs[param_key.replace('estimator__', '')] = search_para[param_key]['dtype'](
                        param_value)
        if 'optimizer' in estimator_kwargs and 'lr' in estimator_kwargs:
            estimator_kwargs['optimizer'] = estimator_kwargs['optimizer'](lr=estimator_kwargs['lr'])
            estimator_kwargs.pop('lr')
        scaler = set_feature_scaler(scaler_dict['name'], **scaler_dict['kwargs'])
        estimator = set_estimator(estimator_dict['name'], **estimator_dict['kwargs'])
        estimator = set_estimation_pipeline(scaler, estimator)
        estimator = self.pure_estimation(estimator, x_data, y_data, True, model_file_name, verb, **kwargs['fit_params'])
        return estimator, log







