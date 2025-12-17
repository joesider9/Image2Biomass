import sys
import os
import joblib
import copy
import importlib
import time
import pandas as pd
import numpy as np

print(os.getcwd())
sys.path.append(os.getcwd())
from eforecast.init.initialize import initializer
from eforecast.clustering.clustering_manager import ClusterOrganizer
from eforecast.common_utils.logger import create_logger
from eforecast.training.train_rbfnns_on_cpus import train_rbfnn_on_cpus
from eforecast.training.train_clustrers_on_cpu import train_clusters_on_cpus

path_sys = '/home/smartrue' if not os.path.exists('/media/sider/data') else '/media/sider/data'
path_data =  os.path.join(path_sys, 'Dropbox/current_codes/PycharmProjects/AdmieRP_train/DATA')
path_log = os.path.join(path_sys,'Dropbox/data_transfer')
logger_name = 'fit_cpu_models_logger' if not os.path.exists('/media/sider/data') else 'fit_cpu_models_logger_win'
logger = create_logger(logger_name, path_log, f'{logger_name}.log', 'w')

def fit_on_cpus(static_data, cluster=None, method=None, refit=False):
    if static_data['is_Fuzzy']:
        train_rbfnn_on_cpus(static_data, cluster=cluster, method=method, refit=refit)
        train_clusters_on_cpus(static_data, cluster=cluster, method=method, refit=refit)
    return 'Done'

def process_pipeline(static_data, safe=True, refit=False):
    if safe:
        try:
            if static_data['transfer_learning']:
                static_data_base = joblib.load(static_data['transfer_learning_from']['configuration'])
                static_data['static_data_base'] = static_data_base
            cluster_organizer = ClusterOrganizer(static_data, is_online=False, train=True, refit=False)
            cluster_organizer.update_cluster_folders()
            fit_on_cpus(static_data)
            del static_data
        except Exception as e:
            print(e)
    else:
        if static_data['transfer_learning']:
            static_data_base = joblib.load(static_data['transfer_learning_from']['configuration'])
            static_data['static_data_base'] = static_data_base
        cluster_organizer = ClusterOrganizer(static_data, is_online=False, train=True, refit=False)
        cluster_organizer.update_cluster_folders()
        fit_on_cpus(static_data)
        del static_data


if __name__ == "__main__":
    from configuration.config import config

    # Initialize the system with configuration settings
    static_data = initializer(config())
    process_pipeline(static_data, safe=False, refit=False)