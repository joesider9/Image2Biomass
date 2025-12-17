import os
import pvlib
import pandas as pd
import polars as pl
import numpy as np
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from eforecast.datasets.image_data.dataset_img_creator import DatasetImageCreator
from eforecast.datasets.image_data.dataset_img_stats_creator import DatasetImageStatsCreator
from eforecast.datasets.data_transformations import DataTransformer
from eforecast.datasets.row_data_transformation import RowDataTransformer
from eforecast.datasets.files_manager import FilesManager
from eforecast.datasets.data_feeder import DataFeeder

from eforecast.common_utils.date_utils import sp_index
from eforecast.common_utils.date_utils import last_year_lags
from eforecast.common_utils.dataset_utils import fix_timeseries_dates


class DatasetCreator:
    def __init__(self, static_data, recreate=False, train=True, is_online=False, dates=None):
        self.static_data = static_data
        self.is_online = is_online
        self.train = train
        self.dates = None
        self.data = None
        self.row_transformer = RowDataTransformer(self.static_data)

        self.load_data()
        self.path_data = self.static_data['path_data']
        self.horizon_type = static_data['horizon_type']
        self.nwp_models = static_data['NWP']
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']
        self.transformer = DataTransformer(self.static_data, recreate=recreate, is_online=self.is_online,
                                           train=self.train)
        self.files_manager = FilesManager(self.static_data, is_online=self.is_online, train=self.train)
        self.data_feeder = DataFeeder(self.static_data, online=self.is_online, train=self.train)
        if recreate or is_online:
            self.files_manager.remove_row_data_files()
            self.files_manager.remove_lstm_data_files()
            self.files_manager.remove_nwps()
            self.files_manager.remove_images()
            if not is_online:
                self.files_manager.remove_target_files()

    def load_data(self):
        data = pd.read_csv(
            '/media/sider/data/Dropbox/current_codes/PycharmProjects/Image2Biomass/csiro-biomass/train.csv')
        data['sample_id'] = data['sample_id'].apply(lambda x: x.split('_')[0])
        data.set_index('sample_id', inplace=True)
        columns = data.columns
        data_temp = data[['target_name', 'target']].reset_index()
        data_expand = data_temp.reset_index().pivot_table(
            index=['sample_id'],
            columns='target_name',
            values='target',
            fill_value=0,
            aggfunc='first'
        )
        data_temp = data[[col for col in columns if col not in ['target_name', 'target']]]
        data_temp = data_temp[~data_temp.index.duplicated(keep='first')]
        self.data = pd.concat([data_temp, data_expand], axis=1)
        self.data = self.row_transformer.transform(self.data)
        self.dates = self.data.index
        print(f"Time series imported successfully from the file {self.static_data['filename']}")

    def create_image_dataset(self, parallel):
        if self.static_data['use_image'] is not None:
            image_stats_creator = DatasetImageStatsCreator(self.static_data, self.transformer, dates=self.dates,
                                                     train=self.train, parallel=parallel)
            image_dates, image_stats = self.files_manager.check_if_exists_image_data()
            if image_dates is None:
                image_dates = [d.split('.')[0] for d in (os.listdir(os.path.join(image_stats_creator.path_sat, 'train'
                if self.train else 'test')))]
                image_stats = image_stats_creator.make_dataset()
                self.files_manager.save_images(image_dates, image_stats)

    def get_data(self, lag, var_data, data, data_pl):
        data_temp = []
        var_name = var_data['name']
        if var_data['source'] == 'target':
            col = self.static_data['project_name']
            if col not in data.columns:
                col = [c for c in self.data.columns if c in var_data['name']][0]
        elif var_data['source'] in {'nwp_dataset', 'index', 'grib'}:
            col = var_name
        elif var_data['source'].endswith('.csv'):
            if var_name in data.columns:
                col = var_name
            else:
                col = data.columns[0]
        else:
            col = var_data['source']
        if isinstance(lag, float):
            raise ValueError('Lag must be integer or string')
        freq = self.static_data['ts_resolution']
        if isinstance(lag, int) or isinstance(lag, np.integer):
            if self.static_data['use_polars']:
                data_temp = data_pl.select(col).shift(-lag)
                data_temp.columns = [var_name]
            else:
                data_temp = data[col].shift(-lag).to_frame()
                data_temp.columns = [var_name]

        return data_temp

    def concat_lagged_data(self, data, var_name, var_data, lstm_lags=None):
        if isinstance(data, pd.Series):
            data = data.to_frame()
        if lstm_lags is not None:
            var_lags = []
            for l in lstm_lags:
                if isinstance(l, int) or isinstance(l, np.integer):
                    if l < 0 or l in var_data['lags']:
                        var_lags.append(l)
                else:
                    var_lags.append(l)
        else:
            var_lags = var_data['lags']
        if len(var_lags) == 0:
            return None
        data_pl = pl.from_pandas(data)
        results = [self.get_data(lag, var_data, data, data_pl) for lag in var_lags]
        # with Pool(5) as pool:
        #     results = pool.map(partial(self.get_data, var_data=var_data, data=data, data_pl=data_pl), var_lags)
        if self.static_data['use_polars']:
            data_temp = pl.concat(list(results), how='horizontal')
            data_temp = data_temp.to_pandas()
        else:
            data_temp = pd.concat(results, axis=1)
        data_temp.index = data.index


        if var_data['use_diff_between_lags']:
            diff_df = []
            for lag1 in var_lags:
                for lag2 in var_lags:
                    if isinstance(lag1, str) or isinstance(lag2, str):
                        continue
                    if np.abs(lag1) > 3 or np.abs(lag2) > 3:
                        continue
                    if lag1 > lag2:
                        diff = data_temp[f'{var_name}_lag_{lag1}'] - data_temp[f'{var_name}_lag_{lag2}']
                        diff = diff.to_frame(f'Diff_{var_name}_lag{lag1}_lag{lag2}')
                        diff2 = np.square(diff)
                        diff2.columns = [f'Diff2_{var_name}_lag{lag1}_lag{lag2}']
                        diff_df.append(pd.concat([diff, diff2], axis=1))
            data_temp = pd.concat([data_temp] + diff_df, axis=1)
        if var_data['transformer'] == 'month':
            data_temp = pd.DataFrame(pd.DatetimeIndex([d[0] for d in data_temp.apply(lambda x: pd.to_datetime(x, format='%Y/%m/%d')).values]).month, index=data_temp.index, columns=data_temp.columns)
        elif var_data['transformer'] == 'mean':
            data_temp = data_temp.mean(axis=1).to_frame(var_name)
        elif var_data['transformer'] == 'max':
            data_temp = data_temp.max(axis=1).to_frame(var_name)
        elif var_data['transformer'] == 'min':
            data_temp = data_temp.min(axis=1).to_frame(var_name)
        elif var_data['transformer'] == 'median':
            data_temp = data_temp.median(axis=1).to_frame(var_name)
        elif var_data['transformer'] == 'std':
            data_temp = data_temp.std(axis=1).to_frame(var_name)
        elif var_data['transformer'] == 'sum':
            data_temp = data_temp.sum(axis=1).to_frame(var_name)
        else:
            data_temp = self.transformer.transform(data_temp, var_name)
        return data_temp

    def wrap_lagged_data(self, var_name, var_data, data, lag_lstm):
        data_temp = self.concat_lagged_data(data, var_name, var_data[var_name], lstm_lags=lag_lstm)
        if data_temp is None:
            return None
        data_temp = data_temp.dropna(axis='index', how='any')
        return data_temp

    def create_autoregressive_dataset(self, lag_lstm=None):
        variables = dict([(var_data['name'], var_data) for var_data in self.static_data['variables']
                          if var_data['type'] == 'timeseries' and var_data['source'] == 'target'])
        if len(variables) == 0:
            return None
        data_arma = []
        data = self.data.copy(deep=True)
        with ProcessPoolExecutor(self.static_data['n_jobs']) as pool:
            results = pool.map(partial(self.wrap_lagged_data, var_data=variables, data=data, lag_lstm=lag_lstm), list(variables.keys()))
            data_arma = [d for d in results if d is not None]
            data_arma = pd.concat(data_arma, axis=1)
        return data_arma.dropna(axis='index', how='any')

    def create_calendar_dataset(self, lag_lstm=None):
        # sin_transformer = lambda x: np.sin(x / period * 2 * np.pi)
        variables_index = dict([(var_data['name'], var_data) for var_data in self.static_data['variables']
                                if var_data['type'] == 'calendar' and var_data['source'] == 'index'])

        if self.data is not None:
            data = self.data.copy(deep=True)
        else:
            data = None
        if not self.is_online:
            index = data.index
        else:
            if data is not None:
                max_lag = min([min(var_data['lags']) for var_data in self.static_data['variables']])
                index = pd.date_range(data.index[0] + pd.DateOffset(hours=max_lag),
                                      self.dates[-1] + pd.DateOffset(hours=47),
                                      freq=self.static_data['ts_resolution'])
            else:
                if self.static_data['horizon_type'] == 'intra-ahead':
                    index = pd.date_range(self.dates[0], self.dates[-1] + pd.DateOffset(hours=23), freq='h')
                else:
                    index = pd.date_range(self.dates[0], self.dates[-1] + pd.DateOffset(hours=47), freq='h')
        data_temp = pd.DataFrame(index=index)
        for var_name, var_data in variables_index.items():
            if var_name == 'hour':
                values = index.hour.values
                period = 24
            elif var_name == 'month':
                values = index.month.values
                period = 12
            elif var_name == 'dayweek':
                values = index.dayofweek.values
                period = 7
            elif var_name == 'dayofyear':
                values = index.dayofyear.values
                period = 1
            elif var_name == 'sp_index':
                values = [sp_index(d, country=self.static_data['country']) for d in index]
            else:
                raise ValueError(f'Unknown variable {var_name} for index and calendar')

            values = pd.DataFrame(values, columns=[var_name], index=index)
            if len(var_data['lags']) > 1:
                values = self.concat_lagged_data(values, var_name, var_data, lstm_lags=lag_lstm)
            if values is not None:
                data_temp = pd.concat([data_temp, values], axis=1)
        variables_astral = dict([(var_data['name'], var_data) for var_data in self.static_data['variables']
                                 if var_data['type'] == 'calendar' and var_data['source'] == 'astral'])
        if len(variables_astral) == 0 and len(variables_index) == 0:
            return None
        for var_name, var_data in variables_astral.items():
            solpos = pvlib.solarposition.get_solarposition(index, self.static_data['coord'][0]
                                                           , self.static_data['coord'][1])
            if var_name not in {'azimuth', 'zenith'}:
                raise ValueError(f'Unknown variable {var_name} for astral and calendar. '
                                 f'Accepted values are azimuth, zenith')
            data_temp = pd.concat([data_temp, solpos[var_name].to_frame()], axis=1)
        data_temp = data_temp.dropna(axis='index', how='any')
        return data_temp

    def create_nwp_ts_datasets(self, lag_lstm=None):
        if self.static_data['NWP'] is None:
            return None
        data_types = self.static_data['base_data_types']
        if self.static_data['type'] in {'load', 'FA'}:
            compress = 'load'
            variables_nwp = dict([(var_data['name'], var_data) for var_data in self.static_data['variables']
                                  if var_data['type'] == 'timeseries' and var_data['source'] != 'target'
                                  and var_data['source'] == 'nwp_dataset'])
        else:
            compress = 'minimal'
            variables_nwp = dict([(var_data['name'], var_data) for var_data in self.static_data['variables']
                                  if var_data['type'] == 'nwp' and var_data['source'] != 'target'
                                  and var_data['source'] == 'grib'])
        data_nwp, metadata = self.data_feeder.feed_inputs('row_nwp',
                                                          merge=data_types['merge'], compress=compress, get_all=True)
        if self.static_data['ts_resolution'] == 'D':
            data_nwp['row_nwp'] = data_nwp['row_nwp'].resample('D').mean()
        data_row_nwp = pd.DataFrame()
        for var_name, var_data in variables_nwp.items():
            var_names = [name for name in data_nwp['row_nwp'].keys() if f'{var_name}_0' in name] \
                if var_name not in data_nwp['row_nwp'].columns else [var_name]
            for name in var_names:
                data = self.concat_lagged_data(data_nwp['row_nwp'][name], name, var_data, lstm_lags=lag_lstm)
                if data is None:
                    continue
                data = data.dropna(axis='index')
                data_row_nwp = pd.concat([data_row_nwp, data], axis=1)
        data_row_nwp = data_row_nwp.dropna(axis='index')
        return data_row_nwp

    def create_extra_ts_datasets(self, lag_lstm=None):
        variables_extra = dict([(var_data['name'], var_data) for var_data in self.static_data['variables']
                                if var_data['type'] == 'timeseries' and var_data['source'] != 'target'
                                and var_data['source'] != 'nwp_dataset'])

        if len(variables_extra) == 0:
            return None
        data_extra = pd.DataFrame()
        for var_name, var_data in variables_extra.items():
            name = var_name
            if var_data['source'].endswith('.csv'):
                if os.path.exists(var_data['source']):
                    data = pd.read_csv(var_data['source'], index_col=0, header=0, parse_dates=True)
                    if name in data.columns:
                        data = data[name].to_frame()
                    if var_data['transformer'] == 'fillnan':
                        data = self.transformer.transform(data, var_name)
                else:
                    raise ImportError(f"{var_data['source']} does not exists")
            else:
                data = pd.read_csv(self.static_data['filename'], index_col=0, header=0, parse_dates=True)
                if var_data['source'] not in data.columns:
                    raise ValueError(f"{var_data['source']} does not exists in main file columns. "
                                     f"Filename is {self.static_data['filename']}")
                data = data[var_data['source']].to_frame()
            data = self.concat_lagged_data(data, name, var_data, lstm_lags=lag_lstm)
            if data is None:
                continue
            data = data.dropna(axis='index')
            data_extra = pd.concat([data_extra, data], axis=1)
        data_extra = data_extra.dropna(axis='index')
        return data_extra

    def create_row_datasets(self):
        data_row = self.files_manager.check_if_exists_row_data(get_all=True)
        if data_row is None:
            data_arma = self.create_autoregressive_dataset()
            cols_obs = list(data_arma.columns) if data_arma is not None else []
            data_extra = self.create_extra_ts_datasets()
            cols_obs += list(data_extra.columns) if data_extra is not None else []
            data_calendar = self.create_calendar_dataset()
            cols_calendar = list(data_calendar.columns) if data_calendar is not None else []
            dfs = [data_arma, data_extra, data_calendar]
            data_row_all = pd.DataFrame()
            for data in dfs:
                if data is not None:
                    data_row_all = pd.concat([data_row_all, data], axis=1)
            data_row_all = data_row_all.dropna(axis='index', how='any' if not self.is_online else 'all')
            data_row = {'row_obs': data_row_all[cols_obs] if len(cols_obs) > 0 else None,
                        'calendar': data_row_all[cols_calendar] if len(cols_calendar) > 0 else None}
            self.files_manager.save_row_data(data_row)

    def create_target(self):
        variable = self.static_data['target_variable']
        data = self.files_manager.check_if_exists_target()
        if data is None:
            data = self.data.copy(deep=True)
            var_col = variable["source"] if variable["source"] in data.columns else self.static_data['_id']
            if self.static_data['horizon_type'] == 'multi-output':
                data = data[variable['columns']].dropna(axis='index')
            elif self.static_data['horizon_type'] == 'day-ahead' and self.static_data['type'] == 'FA':
                data[variable['columns'][0]] = data[var_col].shift(-1)
                data = data[variable['columns'][0]].dropna(axis='index').to_frame()
            else:
                data[variable['columns'][0]] = data[var_col]
                data = data[variable['columns'][0]].dropna(axis='index').to_frame()

            self.files_manager.save_target(data)

    def merge_rnn_variables(self, dataset, data_df, var_list, var_lags):
        time_merge_variables = self.static_data['time_merge_variables'] if (
                len(self.static_data['time_merge_variables']) > 0) else None
        if time_merge_variables is not None:
            for new_var, value in time_merge_variables.items():
                merge_flag = all([v in var_list for v in value])
                if merge_flag:
                    var_for_merge = pd.DataFrame.from_dict(
                        {var_data['name']: [max([v for v in var_data['lags'] if not isinstance(v, str)]),
                                            min([v for v in var_data['lags'] if not isinstance(v, str)]),
                                            any([isinstance(v, str) for v in var_data['lags']])]
                         for var_data in self.static_data['variables']
                         if var_data['name'] in value}, orient='index').sort_values(0)
                    data_merged = pd.DataFrame()
                    for var, lag_lim in var_for_merge.iterrows():
                        lag_lim = lag_lim.values
                        cols = []
                        cols_new = []
                        for lag in var_lags:
                            if isinstance(lag, int) or isinstance(lag, float):
                                if lag <= lag_lim[0] and lag >= lag_lim[1]:
                                    cols.append(f'{var}_lag_{lag}')
                                    cols_new.append(f'{new_var}_lag_{lag}')
                            else:
                                if lag_lim[2]:
                                    ly_cols = [col for col in data_df.columns if 'ly' in col.split('_lag_')[-1]
                                               and f'{var}_lag' in col]
                                    cols += ly_cols
                                    cols_new += [col.replace(var, new_var) for col in ly_cols]
                        data1 = dataset[var][cols]
                        data1.columns = cols_new
                        data_merged = pd.concat([data1, data_merged], axis=1)
                        del dataset[var]
                    dataset[new_var] = data_merged
                    var_list = [v for v in var_list if v not in value] + [new_var]
        return dataset, set(var_list)

    def get_temporal_data(self, dataset, lags, data_type):
        if data_type == 'autoregressive':
            data_df = self.create_autoregressive_dataset(lag_lstm=lags)
        elif data_type == 'calendar':
            data_df = self.create_calendar_dataset(lag_lstm=lags)
        elif data_type == 'nwp_data':
            data_df = self.create_nwp_ts_datasets(lag_lstm=lags)
        else:
            data_df = self.create_extra_ts_datasets(lag_lstm=lags)

        if data_df is not None:
            var_ts = set([col.split('_lag_')[0] for col in data_df.columns if 'Diff' not in col])
        else:
            var_ts = []

        for var in var_ts:
            cols = []
            for lag in lags:
                if isinstance(lag, int) or isinstance(lag, np.integer) or isinstance(lag, float):
                    cols.append(f'{var}_lag_{lag}')
                else:
                    ly_cols = [col for col in data_df.columns if 'ly' in col.split('_lag_')[-1] and f'{var}_lag' in col]
                    cols += ly_cols
            for col in cols:
                if col not in data_df.columns:
                    data_df = pd.concat([data_df, pd.DataFrame(0, index=data_df.index,
                                                               columns=[col])], axis=1)
            dataset[var] = data_df[cols]

        dataset, var_ts = self.merge_rnn_variables(dataset, data_df, var_ts, lags)
        return dataset, var_ts


    def create_lstm_dataset_custom(self):
        data_row = self.files_manager.check_if_exists_row_data(get_all=True)
        if data_row is None:
            raise ImportError(f'Cannot find row dataset')
        data_row = data_row['row_obs']
        cols_obs = list(data_row.columns)
        variables = []
        variables += list(set([c.split('_ahead')[0] for c in cols_obs if '_ahead' in c]))
        variables += list(set([c.split('_behind')[0] for c in cols_obs if '_behind' in c]))
        variables += list(set([c.split('_ahead_5')[0] for c in cols_obs if '_ahead_5' in c]))
        variables += list(set([c.split('_behind_5')[0] for c in cols_obs if '_behind_5' in c]))
        variables += list(set([c.split('_mean')[0] for c in cols_obs if '_mean' in c]))
        data_lstm = self.files_manager.check_if_exists_lstm_data()
        lags_ahead_more = ['+15', '+30', '+45', '+1h', '+1h15', '+1h30', '+2h', '+3h', '+4h']
        lags_behind_more = ['-15', '-30', '-45', '-1h', '-1h15', '-1h30', '-2h', '-3h', '-4h'][::-1]
        lags_ahead = [1, 2, 3, 4, 5, 6, 7]
        lags_behind = [-7, -6, -5, -4, -3, -2, -1, 0]
        lags_behind_5 = [-19, -17, -15, -13, -11, -9]
        lags_ahead_5 = [9, 11, 13, 15, 17, 19]
        if data_lstm is not None:
            df_names = data_lstm['data']['past'].keys()
            dates_lstm = data_lstm['data']['past'][list(df_names)[0]].index
            ind = np.where(dates_lstm > self.static_data['Evaluation_start'])[0]
            ok = len(ind) > 1000
        if data_lstm is None or not ok:
            metadata = dict()
            data = dict()
            data['future'] = dict()
            data['past'] = dict()
            for vars in variables:
                col_lags_ahead_0 = [f'{vars}_ahead_lag_{l}' for l in lags_ahead]
                col_lags_ahead_5 = [f'{vars}_ahead_5_lag_{l}' for l in lags_ahead_5]
                col_lags_behind_0 = [f'{vars}_behind_lag_{l}' for l in lags_behind]
                col_lags_behind_5 = [f'{vars}_behind_5_lag_{l}' for l in lags_behind_5]
                col_lags_ahead_more = [f'{vars}_mean{l}' for l in lags_ahead_more]
                col_lags_behind_more = [f'{vars}_mean{l}' for l in lags_behind_more]
                cols = col_lags_behind_more + col_lags_behind_5 + col_lags_behind_0 + col_lags_ahead_0 + col_lags_ahead_5 + col_lags_ahead_more
                data['past'][vars] = data_row[cols]

            metadata['groups'] = []
            metadata['future_lags'] = []
            metadata['past_lags'] = lags_behind_more + lags_behind_5 + lags_behind + lags_ahead + lags_ahead_5 + lags_ahead_more
            metadata['past_variables'] = []
            dates = pd.DatetimeIndex([])
            for key, value in data.items():
                for key1, value1 in value.items():
                    if dates.shape[0] == 0:
                        dates = value1.index
                    else:
                        dates = dates.intersection(value1.index)
            metadata['dates'] = dates
            self.files_manager.save_lstm_data(data, metadata)

    def create_lstm_dataset(self):
        data_lstm = self.files_manager.check_if_exists_lstm_data()
        if data_lstm is None:
            metadata = dict()
            data = dict()
            data['future'] = dict()
            data['past'] = dict()

            metadata['groups'] = []
            if 'global_past_lags' not in self.static_data.keys():
                raise ValueError('Cannot find global past lags in static_data. Check input configuration')
            if 'global_future_lags' not in self.static_data.keys():
                raise ValueError('Cannot find global future lags in static_data. Check input configuration')

            if isinstance(self.static_data['global_past_lags'], int):
                past_lags = [-i for i in range(1, self.static_data['global_past_lags'])]
            else:
                past_lags = self.static_data['global_past_lags']

            if isinstance(self.static_data['global_future_lags'], int):
                future_lags = [i for i in range(self.static_data['global_future_lags'])]
            else:
                future_lags = self.static_data['global_future_lags']

            data['past'], vars_list = self.get_temporal_data(data['past'], past_lags, 'autoregressive')
            past_vars = vars_list
            data['future'], vars_list = self.get_temporal_data(data['future'], future_lags, 'autoregressive')
            future_vars = vars_list
            data['past'], vars_list = self.get_temporal_data(data['past'], past_lags, 'nwp_data')
            past_vars = past_vars.union(vars_list)
            data['future'], vars_list = self.get_temporal_data(data['future'], future_lags, 'nwp_data')
            future_vars = future_vars.union(vars_list)
            data['past'], vars_list = self.get_temporal_data(data['past'], past_lags, 'extra')
            past_vars = past_vars.union(vars_list)
            data['future'], vars_list = self.get_temporal_data(data['future'], future_lags, 'extra')
            future_vars = future_vars.union(vars_list)
            data['past'], vars_list = self.get_temporal_data(data['past'], past_lags, 'calendar')
            past_vars = past_vars.union(vars_list)
            data['future'], vars_list = self.get_temporal_data(data['future'], future_lags, 'calendar')
            future_vars = future_vars.union(vars_list)

            metadata['future_lags'] = future_lags
            metadata['past_lags'] = past_lags
            metadata['past_variables'] = past_vars.difference(future_vars)

            dates = pd.DatetimeIndex([])
            for key, value in data.items():
                for key1, value1 in value.items():
                    if dates.shape[0] == 0:
                        dates = value1.index
                    else:
                        dates = dates.intersection(value1.index)
            metadata['dates'] = dates
            self.files_manager.save_lstm_data(data, metadata)
