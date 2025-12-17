"""
Functions for ADMIE.configuration files

AUTHOR: G.SIDERATOS

Date: September 2022
"""
import os
import sys


def find_pycharm_path():
    pycharm_path = '~/'
    return pycharm_path


def config_folders(docker):
    """
    Define the folders for your PC
    - pycharm_path: Path of your codes where the code of projects group is located
    - sys_folder: Path where the model weights are saved
    - nwp_folder: Path where nwp grib files are located e.g. nwp_folder + /ECMWF
    param docker: Runtime environment
    return: dict with pycharm_path, sys_folder, nwp_folder
    """
    folders = dict()
    folders['pycharm_path'] = find_pycharm_path()

    folders['sys_folder'] = '~/models/'
    folders['nwp_folder'] = None
    folders['path_image'] = '~/Image2Biomass/csiro-biomass'
    for folder in ['sys_folder', 'path_image', 'pycharm_path']:
        try:
            assert os.path.exists(folders[folder])
        except AssertionError:
            raise FileNotFoundError(f'{folders[folder]} does not exist')
    return folders


def define_n_jobs():

    jobs = {'n_cpus': 16,
            'n_jobs': 16,  # ALL CPUS
            'n_jobs_rbfnn': 3,
            'n_jobs_lstm': 2,
            'n_jobs_cnn_3d': 2,
            'n_jobs_cnn': 2,
            'n_jobs_mlp': 3,
            'intra_op': 2,
            'n_gpus': 1}
    return jobs


def define_enviroment(RUNTIME_BACKEND):
    if RUNTIME_BACKEND == 'TORCH':
        env_name = 'Image2Biomass'
    else:
        raise ValueError(f'Wrong backend name {RUNTIME_BACKEND}. It should be TORCH')
    path_env = '~/Image2Biomass/.venv/bin'
    return env_name, path_env
