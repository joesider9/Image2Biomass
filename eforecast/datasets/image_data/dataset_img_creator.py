import os

import cv2
import h5py
import joblib
from joblib import Parallel
from joblib import delayed
from tqdm import tqdm
import numpy as np
import pandas as pd
import astral
from astral.sun import sun
from einops import rearrange
from multiprocessing import Pool
from eforecast.common_utils.date_utils import convert_timezone_dates


class DatasetImageCreator:

    def __init__(self, static_data, transformer, dates=None, is_online=False, parallel=False, refit=False):
        self.refit = refit
        self.static_data = static_data
        self.transformer = transformer
        self.is_online = is_online
        self.parallel = parallel
        ts_res = str.lower(static_data['ts_resolution'])
        self.dates = dates
        self.path_sat = static_data['path_image']
        self.path_sat_processed = static_data['path_image']
        self.n_jobs = static_data['n_jobs']
        self.variables = dict([(var_data['name'], var_data) for var_data in static_data['variables']
                               if var_data['type'] == 'image'])
        self.apis = set([var_data['source'] for var_data in self.variables.values()])
        for var in self.variables.keys():
            if var in self.transformer.variables_index.keys():
                self.transformer.fit(np.array([]), var, data_dates=dates)
        print(f"Dataset Image data creation started for project {self.static_data['_id']}")

    def daylight(self, date):
        try:
            l = astral.LocationInfo('Custom Name', 'My Region', self.static_data['local_timezone'],
                                    self.static_data['coord'][0], self.static_data['coord'][1])
            sun_attr = sun(l.observer, date=date, tzinfo=self.static_data['local_timezone'])
            sunrise = pd.to_datetime(sun_attr['dawn'].strftime('%Y%m%d %H:%M'), format='%Y%m%d %H:%M')
            sunset = pd.to_datetime(sun_attr['dusk'].strftime('%Y%m%d %H:%M'), format='%Y%m%d %H:%M')
            if sunrise - pd.DateOffset(hours=3) <= date <= sunset + pd.DateOffset(hours=3):
                return date
            else:
                return None
        except:
            return None

    def remove_night_hours(self, dates):
        try:
            with Pool(self.static_data['n_jobs']) as pool:
                daylight_dates = pool.map(self.daylight, dates)
        except:
            daylight_dates = [self.daylight(date) for date in dates]
        daylight_dates = [d for d in daylight_dates if d is not None]
        dates_new = pd.DatetimeIndex(daylight_dates)
        return dates_new

    def make_dataset(self):
        if not os.path.exists(os.path.join(self.path_sat_processed, 'processed')):
            os.makedirs(os.path.join(self.path_sat_processed, 'processed'))
        dates = dict()
        for api in self.apis:
            for var in self.variables.keys():
                if self.variables[var]['source'] != api:
                    continue
                dates_temp = []
                if not self.parallel:
                    for t in tqdm(self.dates):
                        d = self.stack_sat(t, api, var)
                        if d is None:
                            continue
                        dates_temp.append(d)

                else:
                    data = Parallel(n_jobs=5)(
                        delayed(self.stack_sat)(t, api, var) for t in tqdm(self.dates))
                    dates_temp = [d for d in data if d is not None]
                dates[f'{api}_{var}'] = pd.DatetimeIndex(dates_temp)
        return dates

    def stack_sat(self, t, api, var):
        if not os.path.exists(os.path.join(self.path_sat_processed, 'processed',
                                       f'satellite_{api}_{var}_{t.strftime("%Y_%m_%d__%H_%M")}.pkl')):
            res = self.stack_hourly_sat(t, var)
            x_3d = dict()
            x_3d[var] = dict()
            if res[var] is None:
                return None
            data = res[var]['data']
            if len(data) > 0:
                x_3d[var] = data
            else:
                return None
            joblib.dump(x_3d, os.path.join(self.path_sat_processed, 'processed',
                                           f'satellite_{api}_{var}_{t.strftime("%Y_%m_%d__%H_%M")}.pkl'))
            return t
        else:
            return t

    def stack_hourly_sat(self, t, var):
        x_3d = self.create_inp_variables(t, var)
        return x_3d

    def create_inp_variables(self, t, var_name):
        inp_var = dict()
        variable = self.variables[var_name]
        inp_var[var_name] = self.create_inp_lag(t, variable)
        return inp_var

    def create_inp_lag(self, date, variable):
        inp_var = dict()
        inp_var['dates'] = pd.DatetimeIndex([date])
        dates_sat = pd.DatetimeIndex([date + pd.DateOffset(hours=lag) for lag in variable['lags']])
        try:
            if self.static_data['local_timezone'] != 'UTC':
                dates_sat = convert_timezone_dates(dates_sat,
                                                   timezone1=self.static_data['local_timezone'],
                                                   timezone2='UTC')
        except:
            return None
        inp_lag = []
        date_sat = dates_sat[0]
        for i in range(len(dates_sat)):
            sat = None
            start = date_sat + pd.DateOffset(minutes=10)
            end = date_sat
            while sat is None:
                sat = self.read_sat_h5(start, end, variable) if variable['name'] not in {'ir039_ir108_vis006', 'rgb_snow'} else \
                    self.read_sat_jpeg(end, variable['name'])
                if sat is None:
                    end = end - pd.DateOffset(minutes=15)
                    if (start - end).seconds // 60 >= 55 and not self.is_online:
                        break
                    elif (start - end).seconds // 3600 >= 13 and self.is_online:
                        break
            if sat is None:
                return None
            sat = self.transformer.transform(sat, variable['name'], data_dates = pd.DatetimeIndex([end.round('15min')]))
            if sat is None:
                return None
            inp_lag.append(np.expand_dims(sat, axis=0) if sat.ndim == 3 else sat)
            if i >= len(variable['lags']) - 1:
                break
            date_sat = end + pd.DateOffset(hours=int(variable['lags'][i + 1] - variable['lags'][i]))
        if len(inp_lag) == 0:
            return None
        inp_lag = np.vstack(inp_lag)
        inp_lag = rearrange(inp_lag, 'l c w h -> l w h c')
        inp_var['data'] = np.expand_dims(inp_lag, axis=0)
        return inp_var


    def read_sat_jpeg(self, date, image_band):
        date = date.ceil('15min')
        path_sat = os.path.join(self.path_sat, 'jpeg', f'{date.year}_{date.strftime("%B")}_{date.day}',
                                f'{date.hour}')

        if image_band == 'ir039_ir108_vis006':
            image_temp = []
            for image_type in 'ir039_ir108_vis006'.split('_'):
                file_sat = os.path.join(path_sat, f"HRSEVERI_{image_type}_{date.strftime('%Y%m%dT%H%M')}.jpg")
                if not os.path.exists(file_sat):
                    return None
                a = cv2.imread(file_sat)
                if a is None:
                    return None
                a = np.expand_dims(a[..., -1], axis=-1).astype(np.uint8)
                image_temp.append(a)
            image_temp = np.concatenate(image_temp, axis=-1).astype(np.uint8)
        else:
            file_sat = os.path.join(path_sat, f"HRSEVERI_{image_band}_{date.strftime('%Y%m%dT%H%M')}.jpg")
            if not os.path.exists(file_sat):
                return None
            a = cv2.imread(file_sat)
            if a is None:
                return None
            image_temp = a.astype(np.uint8)

        return rearrange(image_temp, 'w h c -> c w h')

    def read_sat_h5(self, start, end, variable):
        dates = pd.date_range(end - pd.DateOffset(hours=1), start, freq='h').ceil('H').sort_values(ascending=False)
        files = []
        dates_files = []
        for date in dates:
            path_file = os.path.join(self.path_sat, f'{date.year}_{date.strftime("%B")}_{date.day}', f'{date.hour}')
            if not os.path.exists(path_file):
                path_file = os.path.join(os.path.split(self.path_sat)[0],
                                         f'{date.year}_{date.strftime("%B")}_{date.day}', f'{date.hour}')
                if not os.path.exists(path_file):
                    continue
            if variable['name'] == 'Cloud_Mask':
                sat_abbr = 'CLOUD'
            elif variable['name'] in {'RBG', 'Infrared', 'Infrared1', 'Infrared2', 'target_RBG'}:
                sat_abbr = 'IR'
            else:
                raise ValueError('Unknown satellite variable name')
            for image_file in os.listdir(path_file):
                if str.upper(sat_abbr) in str.upper(image_file):
                    date_file = pd.to_datetime(image_file.split('.')[0].split('_')[-1], format='%Y%m%dT%H%M%SZ')
                    if end <= date_file <= start:
                        files.append(os.path.join(path_file, image_file))
                        dates_files.append(date_file)
        if len(files) == 0:
            return None
        else:
            dates_files, index = pd.DatetimeIndex(dates_files).sort_values(return_indexer=True, ascending=False)
            for i in index:
                try:
                    file = files[i]
                    data = h5py.File(file, 'r')
                    return rearrange([data[band][()].astype('float') for band in variable['bands']],
                                     'b w c -> b w c')
                except:
                    pass
        return None
