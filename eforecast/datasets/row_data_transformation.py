import numpy as np
import pandas as pd

from eforecast.common_utils.nwp_utils import get_clear_sky


class RowDataTransformer:
    def __init__(self, static_data):
        self.transformers = dict()
        self.static_data = static_data
        self.target_variable = self.static_data['target_variable']

        self.coord = self.static_data['coord']
        self.local_timezone = self.static_data['local_timezone']
        self.site_timezone = self.static_data['site_timezone']
        self.ts_resolution = self.static_data['ts_resolution']
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']

    def transform(self, data, inverse=False):
        transformation = self.target_variable['transformer'][0] if self.target_variable['transformer'] is not None else None
        if transformation is None:
            return data
        trans_param = self.target_variable['transformer_params'][transformation]
        if transformation == 'pv_clear_sky':
            if isinstance(data, pd.Series):
                data = data.to_frame()
            if isinstance(data, pd.DataFrame):
                dates = data.index
            else:
                raise ValueError('Data should be dataframe')
            ghi = get_clear_sky(dates, self.coord[0], self.coord[1], self.local_timezone, self.site_timezone,
                                    self.ts_resolution)
            dates_diff = dates.difference(ghi.index)
            if dates_diff.shape[0] > 0:
                ghi_new = get_clear_sky(dates_diff, self.coord[0], self.coord[1], self.local_timezone,
                                        self.site_timezone, self.ts_resolution)
                ghi = pd.concat([ghi, ghi_new])
                ghi = ghi.sort_index()
            ghi = ghi[~ghi.index.duplicated()]
            cs_new = ghi.loc[dates]

            rate = 1 - (trans_param * (cs_new - cs_new.min().values[0]) / (cs_new.max().values[0] - cs_new.min().values[0]))
            data_new = []
            for col in data.columns:
                data_temp = data[col].to_frame()
                if not inverse:
                    data_temp = data_temp.multiply(rate.rename(columns={'clear_sky': col}))
                else:
                    rate_tr = 1 / rate
                    data_temp = data_temp.multiply(rate_tr.rename(columns={'clear_sky': col}))
                data_temp[data_temp < 0] = 0
                data_temp = data_temp.astype(np.float32)
                data_new.append(data_temp)
            data = pd.concat(data_new, axis=1)
        else:
            raise NotImplementedError(f'{transformation} transformation is not implemented yet')
        return data

