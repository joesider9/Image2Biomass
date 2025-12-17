import os
import numpy as np
from configuration.config_project import config_project
from configuration.config_utils import *

static_data = config_project()
path_owner = os.path.join(static_data['sys_folder'], static_data['project_owner'])
path_data = os.path.join(path_owner, f"{static_data['projects_group']}_ver{static_data['version_group']}", 'DATA')

NWP_MODELS = static_data['NWP']
NWP = NWP_MODELS is None

TYPE = static_data['type']
ts_resolution = 1

NWP_DATA_MERGE = [None]  # 'all', 'by_area', 'by_area_variable',#! THIS IS NOT EMPTY. SET [None] INSTEAD
# 'by_variable',#! THIS IS NOT EMPTY. SET [None] INSTEAD
# by_nwp_provider
#! THIS IS NOT EMPTY. SET [None] INSTEAD
DATA_COMPRESS = [None]  # dense or semi_full or full or load#! THIS IS NOT EMPTY. SET [None] INSTEAD
DATA_IMG = ['csiro-biomass']#! THIS IS NOT EMPTY. SET [None] INSTEAD

DATA_IMG_SCALE = ['minmax'] #'minmax', 'standard', 'maxabs'#! THIS IS NOT EMPTY. SET [None] INSTEAD
DATA_NWP_SCALE = [None] #'minmax', 'standard', 'maxabs'#! THIS IS NOT EMPTY. SET [None] INSTEAD
DATA_ROW_SCALE = ['minmax'] #'minmax', 'standard', 'maxabs'#! THIS IS NOT EMPTY. SET [None] INSTEAD
#! THIS IS NOT EMPTY. SET [None] INSTEAD
DATA_TARGET_SCALE = 'minmax' #'minmax', 'standard', 'maxabs'#! THIS IS NOT EMPTY. SET [None] INSTEAD
#! THIS IS NOT EMPTY. SET [None] INSTEAD
USE_DATA_BEFORE_AND_AFTER_TARGET = False

REMOVE_NIGHT_HOURS = False

USE_POLARS = False

HORIZON = static_data['horizon']
HORIZON_TYPE = static_data['horizon_type']

## TRANSFORMER FEATURES

GLOBAL_PAST_LAGS = [-int(i) for i in range(1, int(3 / ts_resolution))]
GLOBAL_FUTURE_LAGS = [int(i) for i in range(int(HORIZON / ts_resolution))]

TIME_MERGE_VARIABLES = {}
if HORIZON_TYPE == 'multi-output':
    targ_lags = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g',
       'Dry_Total_g', 'GDM_g']
else:
    targ_lags = [0]
targ_tag = 'Step' if ts_resolution == 0.25 else 'Hour'

TARGET_VARIABLE = {'name': ['biomass'],
                   'source': 'biomass',
                   'lags': targ_lags,
                   'columns': ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g','Dry_Total_g', 'GDM_g'],
                   'transformer': None,
                   'transformer_params': None
}
## LAGs for NWP and Images are hourly steps

def variables():
    if TYPE == 'Biomass':
        # Labels for NWP variables: Flux, Cloud, Temperature
        sat_inputs = [
            variable_wrapper('csiro-biomass',
                             input_type='image', source='kaggle',
                             ),
           ]
        variable_list = sat_inputs + [
            # variable_wrapper('State', input_type='timeseries', source='target', lags=[0],
            #                  transformer='label_encode',
            #                  transformer_params=[(0, 'Tas'), (1, 'NSW'), (2, 'WA'), (3, 'Vic')]),
            # variable_wrapper('Species', input_type='timeseries', source='target', lags=[0],
            #                  transformer='label_encode',
            #                  transformer_params=[(0, 'Ryegrass_Clover'), (1, 'Lucerne'), (2, 'SubcloverDalkeith'),
            #                                      (3, 'Ryegrass'), (4, 'Phalaris_Clover'), (5, 'SubcloverLosa'),
            #                                      (6, 'Clover'), (7, 'Fescue_CrumbWeed'), (8, 'Phalaris_Ryegrass_Clover'),
            #                                      (9, 'Phalaris'), (10, 'WhiteClover'), (11, 'Fescue'),
            #                                      (12, 'Phalaris_BarleyGrass_SilverGrass_SpearGrass_Clover_Capeweed'),
            #                                      (13, 'Phalaris_Clover_Ryegrass_Barleygrass_Bromegrass'), (14, 'Mixed')]),
            # variable_wrapper('Pre_GSHH_NDVI', input_type='timeseries', source='target', lags=[0]),
            # variable_wrapper('Height_Ave_cm', input_type='timeseries', source='target', lags=[0]),
            # variable_wrapper('Sampling_Date', input_type='timeseries', source='target', lags=[0], transformer='month')
        ]
    else:
        raise NotImplementedError(f'Define variables for type {TYPE}')
    return variable_list


def variable_wrapper(name, input_type='nwp', source='grib', lags=None, timezone='UTC', nwp_provider=None,
                     transformer=None, transformer_params=None, bands=None, use_diff_between_lags=False):
    if nwp_provider is not None:
        if nwp_provider == 'ALL':
            providers = [nwp_model['model'] for nwp_model in NWP_MODELS]
        else:
            providers = [nwp_model['model'] for nwp_model in NWP_MODELS if nwp_model['model'] == nwp_provider]
    else:
        providers = None

    return {'name': name,
            'type': input_type,  # nwp or timeseries or calendar
            'source': source,  # use 'target' for the main timeseries otherwise 'grib', 'database' for nwps,
            # 'nwp_dataset' to get data from created nwp dataset,
            # a column label of input file csv or a csv file extra, 'index' for calendar variables,
            # 'astral' for zenith, azimuth
            'lags': define_variable_lags(name, input_type, lags),
            'timezone': timezone,
            'transformer': transformer,
            'transformer_params': transformer_params,
            'bands': bands,
            'nwp_provider': providers,
            'use_diff_between_lags': use_diff_between_lags
            }


def define_variable_lags(name, input_type, lags):
    if lags is None or lags == 0:
        lags = [0] if HORIZON_TYPE != 'multi-output' else [int(i) for i in range(int(HORIZON / ts_resolution))]
    elif isinstance(lags, int):
        lags = [-int(i) for i in range(int(lags / ts_resolution))]
    elif isinstance(lags, list):
        pass
    else:
        raise ValueError(f'lags should be None or int or list')
    if name in {'Flux', 'wind'}:
        if USE_DATA_BEFORE_AND_AFTER_TARGET:
            if HORIZON == 0:
                max_lag = np.max(lags)
                min_lag = np.min(lags)
                lags = [min_lag - 1] + lags + [max_lag + 1]
    return lags


def config_data():
    static_input_data = {'nwp_data_merge': NWP_DATA_MERGE,
                         'compress_data': DATA_COMPRESS,
                         'img_data': DATA_IMG,
                         'use_data_before_and_after_target': USE_DATA_BEFORE_AND_AFTER_TARGET,
                         'remove_night_hours': REMOVE_NIGHT_HOURS,
                         'variables': variables(),
                         'target_variable': TARGET_VARIABLE,
                         'time_merge_variables': TIME_MERGE_VARIABLES,
                         'global_past_lags': GLOBAL_PAST_LAGS,
                         'global_future_lags': GLOBAL_FUTURE_LAGS,
                         'scale_row_method': DATA_ROW_SCALE,
                         'scale_img_method': DATA_IMG_SCALE,
                         'scale_nwp_method': DATA_NWP_SCALE,
                         'scale_target_method': DATA_TARGET_SCALE,
                         'use_polars': USE_POLARS
                         }
    return static_input_data
