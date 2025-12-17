import os
import joblib

import numpy as np
import pandas as pd

from eforecast.common_utils.dataset_utils import sync_data_with_dates
from eforecast.common_utils.dataset_utils import load_data_shallow_models
from eforecast.datasets.files_manager import FilesManager
from eforecast.datasets.data_feeder import DataFeeder

import eforecast.feature_selection.arfs.feature_selection.allrelevant as arfsgroot

import lightgbm as lgb
from boruta import BorutaPy
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import MultiTaskLassoCV, MultiTaskLasso
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from catboost import Pool
import itertools

CategoricalFeatures = ['dayweek', 'hour', 'month', 'sp_index']


class FeatureSelector:
    def __init__(self, static_data, recreate=False, online=False, train=False):
        self.feature_selectors = dict()
        self.online = online
        self.train = train
        self.static_data = static_data
        self.is_Fuzzy = self.static_data['is_Fuzzy']
        self.clusters = dict()
        if self.is_Fuzzy:
            self.clusters = joblib.load(os.path.join(static_data['path_model'], 'clusters.pickle'))
        if any(list(static_data['global_methods'].values())):
            cluster_path = os.path.join(static_data['path_model'], 'global')
            if not os.path.exists(cluster_path):
                os.makedirs(cluster_path)
            self.clusters.update({'global': cluster_path})
        self.recreate = recreate
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']

        self.dataset_tags = set([tag for exp in self.static_data['experiments'].values() for tag in exp.keys()
                              if tag not in {'output', 'hidden_layer'}])
        self.feature_selection_methods = self.static_data['feature_selection_methods']
        self.calendar_variables = [var_data['name'] for var_data in self.static_data['variables']
                                   if var_data['type'] == 'calendar']
        self.rated = static_data['rated']
        self.files_manager = FilesManager(static_data, is_online=online)
        self.data_feeder = DataFeeder(self.static_data, online=self.online, train=self.train)

    def estimator(self, method, alpha=None, multi_output=False, size=50000, m_estimator=1):
        loss_function = 'MultiRMSE' if self.static_data['horizon_type'] == 'multi-output' else 'RMSE'
        model = CatBoostRegressor(loss_function=loss_function, allow_writing_files=False, silent=True)
        if method.split('_')[0] == 'Leshy':
            return arfsgroot.Leshy(
                model, n_estimators=100, verbose=0, max_iter=20, random_state=42, importance=method.split('_')[1]
            )
        elif method.split('_')[0] == 'BoostAGroota':
            return arfsgroot.BoostAGroota(estimator=model, cutoff=1, iters=20, max_rounds=10, delta=0.1,
                                          importance=method.split('_')[1])
        elif method == 'lasso':
            if alpha is None:
                if multi_output:
                    return MultiTaskLassoCV(max_iter=150000, n_jobs=self.static_data['n_jobs'])
                else:
                    return LassoCV(max_iter=150000, positive=True, n_jobs=self.static_data['n_jobs'])
            else:
                if multi_output:
                    return MultiTaskLasso(max_iter=150000, alpha=alpha)
                else:
                    return Lasso(max_iter=150000, positive=True, alpha=alpha)
        elif method == 'FeatureImportance':
            return RandomForestRegressor(n_estimators=int(m_estimator * 200), max_samples=size, n_jobs=self.static_data['n_jobs'])
        elif method == 'boruta':
            subsample =  1 / m_estimator if m_estimator > 1 else 1.
            n_estimators = int(m_estimator * 200) if m_estimator > 1 else 500
            if multi_output:
                selector = dict()
                for col in self.static_data['target_variable']['lags']:
                    est = lgb.LGBMRegressor(n_estimators=n_estimators, n_jobs=self.static_data['n_jobs'], subsample=subsample, verbose=0)
                    selector[f'boruta_{col}'] = BorutaPy(est, n_estimators=200, verbose=0)
            else:
                selector = BorutaPy(lgb.LGBMRegressor(n_estimators=n_estimators, n_jobs=self.static_data['n_jobs'],
                                                      subsample=subsample, verbose=0),
                                    n_estimators=n_estimators, max_iter=32, verbose=0)
            return selector
        elif method == 'ShapValues':
            loss_function = 'MultiRMSE' if self.static_data['horizon_type'] == 'multi-output' else 'RMSE'
            return CatBoostRegressor(iterations=200, loss_function=loss_function, allow_writing_files=False)
        else:
            raise ValueError(f'Unknown feature selection method {method}')

    @staticmethod
    def importance_vector(method, estimator, train_pool=None):
        if method == 'lasso':
            return np.abs(estimator.coef_)
        elif method == 'FeatureImportance':
            return estimator.feature_importances_
        elif method == 'boruta':
            if isinstance(estimator, dict):
                hist = None
                for col in estimator.keys():
                    if hist is None:
                        hist = estimator[col].importance_history_.sum(axis=0)
                    else:
                        hist += estimator[col].importance_history_.sum(axis=0)
            else:
                hist = estimator.importance_history_.sum(axis=0)
            return hist
        elif method == 'ShapValues':
            fi = estimator.get_feature_importance(data=train_pool,
                                                  type=method,
                                                  prettified=True)
            if isinstance(fi, np.ndarray):
                return np.mean(np.mean(np.abs(fi), axis=1), axis=0)[:-1]
            else:
                return fi.abs().mean(axis=0).values[:-1]
        else:
            raise ValueError(f'Unknown feature selection method {method}')

    def fit_catboost(self, selector, x_train, y_train, cols, cat_feats):
        try:
            selector.fit(x_train, y_train[cols], cat_features=cat_feats, verbose=False)
        except:
            if self.static_data['horizon_type'] == 'multi-output':
                selector = self.estimator('ShapValues', multi_output=True)
                selector.fit(x_train, y_train[cols] +
                             pd.DataFrame(np.random.uniform(0, 0.0001, list(y_train[cols].shape)),
                                          index=y_train.index, columns=cols),
                             cat_features=cat_feats, verbose=False)
            else:
                raise ValueError('Cannot fit Catboost')
        return selector

    def fit_boruta(self, selector, x_train, y_train):
        if isinstance(selector, dict):
            for col in self.static_data['target_variable']['lags']:
                selector[f'boruta_{col}'].fit(x_train.values, y_train.iloc[:, col].values)
        else:
            selector.fit(x_train.values, y_train.values.ravel())
        return selector

    def fit_lgbm(self, x_train, y_train, multi_output, m_estimator):
        subsample = 1 / m_estimator if m_estimator > 1 else 1.
        n_estimators = int(m_estimator * 200) if m_estimator > 1 else 200
        if multi_output:
            estimator = dict()
            x_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
            for col in self.static_data['target_variable']['lags']:
                lgbc_fit_params = {
                    'eval_set': (X_val, y_val[:, col]),
                    'eval_metric': 'l1'
                }
                estimator[f'lgbm_{col}'] = lgb.LGBMRegressor(n_estimators=n_estimators, n_jobs=self.static_data['n_jobs'],
                                                             subsample=subsample, verbose=0)
                estimator[f'lgbm_{col}'].fit(x_train.values, y_train.values[:, col], **lgbc_fit_params)
        else:
            x_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
            lgbc_fit_params = {
                'eval_set': (X_val, y_val),
                'eval_metric': 'l1'
            }
            estimator = lgb.LGBMRegressor(n_estimators=n_estimators, n_jobs=self.static_data['n_jobs'],
                                          subsample=subsample, verbose=0)
            estimator.fit(x_train.values, y_train.values.ravel(), **lgbc_fit_params)
        return estimator

    def predict_lgbm(self, estimator, x_test):
        if isinstance(estimator, dict):
            pred = []
            for col in self.static_data['target_variable']['lags']:
                pred.append(estimator[f'lgbm_{col}'].predict(x_test.values))
            pred = np.concatenate(pred, axis=1)
        else:
            pred = estimator.predict(x_test.values)
        return pred

    def fit_method(self, method, x_train, y_train, x_test, y_test, cat_feats=None):
        if y_train.shape[1] > 1:
            cols = y_train.columns
            multi_output = True
        else:
            cols = y_train.columns[0]
            multi_output = False
        feature_selector = dict()
        thresholds = np.logspace(-6, -1, 6).tolist() + [1]
        selector = self.estimator(method, multi_output=multi_output, size=x_train.shape[0] - 1
                                                                            if x_train.shape[0] < 100000
                                                                                else 100000, m_estimator=1
                                                                                                        if x_train.shape[0] < 100000
                                                                                                        else x_train.shape[0] / 100000)
        if cat_feats is not None:
            cat_feats = list(set([v_name for v_name in x_train.columns
                         for c_feats in CategoricalFeatures if c_feats in v_name]))
            x_train[cat_feats] = x_train[cat_feats].astype('int')
        if method == 'ShapValues':
            selector = self.fit_catboost(selector, x_train, y_train, cols, cat_feats)
        elif method == 'boruta':
            selector = self.fit_boruta(selector, x_train, y_train[cols])
        else:
            selector.fit(x_train, y_train[cols])
        if method == 'lasso':
            alpha = selector.alpha_
        else:
            alpha = None
        mae = []
        importance = self.importance_vector(method, selector, Pool(x_train, y_train[cols], cat_features=cat_feats))
        importance = (importance - np.min(importance)) / (np.max(importance) - np.min(importance))
        if len(importance.shape) > 1 and multi_output:
            importance = np.sum(importance, axis=0)
        for threshold in thresholds:
            indices = np.where(importance > threshold)[0]
            if indices.shape[0] > 4:
                names = x_train.columns[indices]
                selector_temp = self.estimator(method, alpha=alpha, multi_output=multi_output)
                if cat_feats is not None:
                    cat_feats = list(set([v_name for v_name in names for c_feats in CategoricalFeatures
                                          if c_feats in v_name]))
                    x_train[cat_feats] = x_train[cat_feats].astype('int')
                    x_test[cat_feats] = x_test[cat_feats].astype('int')
                if method == 'ShapValues':
                    selector_temp = self.fit_catboost(selector_temp, x_train[names], y_train, cols, cat_feats)
                    pred = selector_temp.predict(Pool(x_test[names], cat_features=cat_feats))
                elif method == 'boruta':
                    selector_temp = self.fit_lgbm(x_train[names], y_train[cols], multi_output, m_estimator=1
                                                                                                        if x_train.shape[0] < 100000
                                                                                                        else x_train.shape[0] / 100000)
                    pred = self.predict_lgbm(selector_temp, x_test[names])
                else:
                    selector_temp.fit(x_train[names], y_train[cols])
                    pred = selector_temp.predict(x_test[names])
                if self.rated is not None:
                    mae.append(np.mean(np.abs(pred - y_test[cols].values)))
                else:
                    mae.append(np.mean(np.abs(pred - y_test[cols].values) / y_test[cols].values))
            else:
                mae.append(np.inf)
        if np.all(np.isinf(mae)):
            best_threshold = np.mean(importance)
        else:
            best_threshold = thresholds[np.argmin(mae)]
        feature_indices = np.where(importance > best_threshold)[0]
        if feature_indices.shape[0] < 4:
            feature_indices = np.arange(importance.shape[0])
        feature_names = x_train.columns[feature_indices]
        feature_selector['indices'] = feature_indices
        feature_selector['names'] = feature_names
        return feature_selector

    def fit_method_new(self, method, x_train, y_train, x_test, y_test, cat_feats=None):
        if y_train.shape[1] > 1:
            multi_output = True
        else:
            multi_output = False
        feature_selector = dict()
        feature_indices = np.array([i for i in range(x_train.shape[1])])
        selector = self.estimator(method, multi_output=multi_output)

        selector.fit(x_train, y_train)

        feature_indices = feature_indices[selector.support_]
        if len(feature_indices) < 2:
            feature_indices = ([i for i in range(x_train.shape[1])])
        feature_selector['indices'] = feature_indices
        feature_selector['names'] = selector.selected_features_
        return feature_selector

    def fit_method_lstm(self, method, x_train, y_train, x_test, y_test, metadata):
        feature_selector = dict()
        x_train_, x_test_ = self.compress_lstm(x_train, x_test, axis=1)
        feature_selector_ = self.fit_method_on_compressed(method, x_train_, y_train, x_test_, y_test)
        ind_lags = feature_selector_['indices']
        x_train_, x_test_ = self.compress_lstm(x_train[:, ind_lags, :], x_test[:, ind_lags, :], axis=2)
        feature_selector_ = self.fit_method_on_compressed(method, x_train_, y_train, x_test_, y_test,
                                                          columns=metadata['variables'])
        ind_vars = feature_selector_['indices']

        feature_selector['lags'] = ind_lags
        feature_selector['variables'] = [metadata['variables'][i] for i in ind_vars]
        return feature_selector

    @staticmethod
    def compress_lstm(x, x_test, axis=2):
        x_compress = None
        x_compress_test = None
        for var in range(x.shape[axis]):
            if axis == 1:
                X = x[:, var, :]
                X_test = x_test[:, var, :]
            elif axis == 2:
                X = x[:, :, var]
                X_test = x_test[:, :, var]
            else:
                raise ValueError("Axis parameter should be 1 or 2")
            X1 = np.concatenate([X, X_test])
            m = MLPRegressor(activation='identity', hidden_layer_sizes=(1,), max_iter=1000).fit(X1, X1)
            x_ = np.matmul(X, m.coefs_[0])
            x_test_ = np.matmul(X_test, m.coefs_[0])
            x_compress = np.concatenate([x_compress, x_], axis=1) if x_compress is not None else x_
            x_compress_test = np.concatenate([x_compress_test, x_test_], axis=1) \
                if x_compress_test is not None else x_test_
        return x_compress, x_compress_test

    def fit_method_on_compressed(self, method, x_train, y_train, x_test, y_test, columns=None):
        if columns is None:
            cols = [f'col_{i}' for i in range(x_train.shape[1])]
        else:
            cols = columns
        x_train = pd.DataFrame(x_train, columns=cols)
        x_test = pd.DataFrame(x_test, columns=cols)
        if method.split('_')[0] in {'Leshy', 'BoostAGroota'}:
            feature_selector_ = self.fit_method_new(method, x_train, y_train, x_test, y_test)
        else:
            cat_feats = [col for col in x_train.columns if x_train[col].unique().shape[0] < 30]
            x_train[cat_feats] = x_train[cat_feats].astype('int')
            x_test[cat_feats] = x_test[cat_feats].astype('int')
            feature_selector_ = self.fit_method(method, x_train, y_train, x_test, y_test)
        return feature_selector_


    @staticmethod
    def concat_lstm(x, metadata, dates):
        dates = metadata['dates'].intersection(dates)
        x_new = []
        var_names = []
        for var_name in sorted(x['past'].keys()):
            past = np.expand_dims(x['past'][var_name].loc[dates].values, axis=-1)
            if var_name in x['future'].keys():
                future = np.expand_dims(x['future'][var_name].loc[dates].values, axis=-1)
            else:
                future = np.zeros([past.shape[0], len(metadata['future_lags']), 1])
            x_new.append(np.concatenate([past, future], axis=1))
            var_names.append(var_name)

        return np.concatenate(x_new, axis=-1), var_names

    def _fit(self, x, y, cv_mask, fs_selector_name, method, metadata=None):
        if 'lstm' in fs_selector_name:
            x_train, _= self.concat_lstm(x, metadata, cv_mask[0].union(cv_mask[1]))
            y_train = sync_data_with_dates(y, cv_mask[0].union(cv_mask[1]))
            x_test, var_names= self.concat_lstm(x, metadata, cv_mask[2])
            y_test = sync_data_with_dates(y, cv_mask[2])
            metadata['variables'] = var_names
        else:
            x_train = sync_data_with_dates(x, cv_mask[0].union(cv_mask[1]))
            y_train = sync_data_with_dates(y, cv_mask[0].union(cv_mask[1]))

            x_test = sync_data_with_dates(x, cv_mask[2])
            y_test = sync_data_with_dates(y, cv_mask[2])



        print(f'Fitting {fs_selector_name}')
        if 'lstm' in fs_selector_name:
            feature_selector = self.fit_method_lstm(method, x_train, y_train, x_test, y_test, metadata)
        else:
            if isinstance(x_train, pd.DataFrame):
                if method.split('_')[0] in {'Leshy', 'BoostAGroota'}:
                    feature_selector = self.fit_method_new(method, x_train, y_train, x_test, y_test)
                else:
                    feature_selector = self.fit_method(method, x_train, y_train, x_test, y_test)
            else:
                raise ValueError('Cannot recognize action for feature selection')

        return feature_selector

    def rename_features(self, dict, old_name, new_name):
        for k, v in dict.items():
            if 'variables' in v.keys():
                dict[k]['variables'] = [n.replace(old_name, new_name) for n in v['variables'] if old_name in n]
            elif 'names' in v.keys():
                dict[k]['names'] = np.array([n.replace(old_name, new_name) for n in list(v['names']) if old_name in n])
            else:
                dict[k] = self.rename_features(v, old_name, new_name)
        return dict

    def transfer_learning(self, static_data_base):
        file_backup = os.path.join(static_data_base['path_model'], 'feature_selectors.pickle')
        if os.path.exists(file_backup):
            feature_selectors = joblib.load(file_backup)
            feature_selectors = self.rename_features(feature_selectors, static_data_base['project_name'],
                                                     self.static_data['project_name'])
            joblib.dump(feature_selectors, os.path.join(self.static_data['path_model'], 'feature_selectors.pickle'))
            for cluster_name, cluster_path in self.clusters.items():
                filename = os.path.join(cluster_path, 'feature_selectors.pickle')
                joblib.dump(feature_selectors[cluster_name], filename)
        else:
            clusters = joblib.load(os.path.join(static_data_base['path_model'], 'clusters.pickle'))
            cluster_path = os.path.join(static_data_base['path_model'], 'global')
            clusters.update({'global': cluster_path})
            for cluster_name, cluster_path in clusters.items():
                filename = os.path.join(cluster_path, 'feature_selectors.pickle')
                fs = joblib.load(filename)
                fs = self.rename_features(fs, static_data_base['project_name'],
                                                         self.static_data['project_name'])
                joblib.dump(fs, os.path.join(clusters[cluster_name], 'feature_selectors.pickle'))

    def create_backup(self):
        file_backup = os.path.join(self.static_data['path_model'], 'feature_selectors.pickle')
        feature_selectors = dict()
        for cluster_name, cluster_path in self.clusters.items():
            filename = os.path.join(cluster_path, 'feature_selectors.pickle')
            feature_selectors.update({cluster_name: joblib.load(filename)})
        joblib.dump(feature_selectors, file_backup)

    def fit(self, recreate_lstm=False):
        for cluster_name, cluster_path in self.clusters.items():
            feature_selectors = dict()
            filename = os.path.join(cluster_path, 'feature_selectors.pickle')
            if os.path.exists(filename):
                if self.recreate:
                    pass
                else:
                    feature_selectors.update(joblib.load(filename))
                    if recreate_lstm:
                        feature_selectors = {k:v for k, v in feature_selectors.items() if 'lstm' not in k}
            if cluster_name == 'global':
                cv_mask = self.files_manager.check_if_exists_cv_data()
            else:
                cv_mask = joblib.load(os.path.join(cluster_path, 'cv_mask.pickle'))
            for method in self.feature_selection_methods:
                if method is not None and 'lstm' in self.dataset_tags:
                    for scale_row_method in self.static_data['scale_row_method']:
                        dataset_name = f'{scale_row_method}_lstm'
                        fs_selector_name = f'feature_selector_{cluster_name}_{method}_{dataset_name}'
                        if fs_selector_name not in feature_selectors.keys():
                            data_type = {'lstm': {'scale_row_method': scale_row_method}}
                            x, y, metadata = load_data_shallow_models(self.data_feeder, data_type,
                                                                      'lstm', True,
                                                                      get_lstm_vars=True)
                            feature_selectors[fs_selector_name] = self._fit(x['lstm'], y, cv_mask, fs_selector_name,
                                                                            method,
                                                                            metadata=metadata['lstm'])
                            self.save(cluster_path, feature_selectors)
            for data_tag in self.dataset_tags:
                data_types = []
                if 'row' in data_tag:
                    if 'all' in data_tag or 'nwp' in data_tag or 'stats' in data_tag:
                        data_types.extend([{data_tag: {'scale_row_method': scale_row_method,
                                            'scale_nwp_method': scale_nwp_method,
                                            'scale_img_method': scale_img_method,
                                            'merge': merge,
                                            'compress': compress,
                                            'image_data_type': image_data_type
                                            }}
                                           for scale_row_method in self.static_data['scale_row_method']
                                           for scale_nwp_method in self.static_data['scale_nwp_method']
                                           for scale_img_method in self.static_data['scale_img_method']
                                           for merge in self.static_data['nwp_data_merge']
                                           for compress in self.static_data['compress_data']
                                           for image_data_type in self.static_data['img_data']])
                    else:
                        data_types.extend([{data_tag: {'scale_row_method': scale_row_method,
                                                            }}
                                                for scale_row_method in self.static_data['scale_row_method']])

                for data_type in data_types:
                    for method in self.feature_selection_methods:
                        if method is not None:
                            x, y, metadata = load_data_shallow_models(self.data_feeder, data_type, data_tag,
                                                                      True)
                            if not isinstance(x[data_tag], dict):
                                dataset_name = ('_'.join([t for t in data_type[data_tag].values() if t is not None]) +
                                                f'_{data_tag}')
                                fs_selector_name = f'feature_selector_{cluster_name}_{method}_{dataset_name}'
                                if fs_selector_name not in feature_selectors.keys():
                                    feature_selectors[fs_selector_name] = self._fit(x[data_tag], y, cv_mask,
                                                                                    fs_selector_name, method,
                                                                                    metadata=metadata[data_tag])
                            else:
                                for group in x[data_tag].keys():
                                    dataset_name = ('_'.join([t for t in data_type[data_tag].values()
                                                              if t is not None]) + f'_{data_tag}_{group}')
                                    fs_selector_name = f'feature_selector_{cluster_name}_{method}_{dataset_name}'
                                    if fs_selector_name not in feature_selectors.keys():
                                        feature_selectors[fs_selector_name] = self._fit(x[data_tag][group], y,
                                                                                        cv_mask, fs_selector_name,
                                                                                        method,
                                                                                        metadata=metadata[data_tag])
                            self.save(cluster_path, feature_selectors)
                self.save(cluster_path, feature_selectors)

    @staticmethod
    def save(cluster_path, feature_selectors):
        filename = os.path.join(cluster_path, 'feature_selectors.pickle')
        joblib.dump(feature_selectors, filename)
