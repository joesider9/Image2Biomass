import sys
import os
import importlib
print(os.getcwd())
sys.path.append(os.getcwd())
from eforecast.init.initialize import initializer
from eforecast.nwp_extraction.nwp_extractor import NwpExtractor

from eforecast.datasets.dataset_creator import DatasetCreator
from eforecast.datasets.data_preprocessing.data_pipeline import DataPipeline
from eforecast.clustering.clustering_manager import ClusterOrganizer
from eforecast.datasets.data_preprocessing.data_split import Splitter
from eforecast.feature_selection.feature_selection_fit import FeatureSelector

import traceback
from eforecast.common_utils.train_utils import send_predictions

RECREATE_DATASETS = False

def nwp_extraction(static_data, dataset):
    nwp_extractor = NwpExtractor(static_data, recreate=RECREATE_DATASETS, is_online=False,
                                 dates=dataset.dates)
    nwp_extractor.extract()


def create_datasets(dataset):

    if static_data['use_image']:
        dataset.create_image_dataset(parallel=True)
    # dataset.create_target()
    # dataset.create_row_datasets()



def preprocess_data(static_data):
    pipeline = DataPipeline(static_data, recreate=True, online=False, train=True)
    pipeline.fit_pipe()


def clustering_fit(static_data):
    splitter = Splitter(static_data, is_online=False, train=True)
    splitter.split(refit=True)
    cluster_organizer = ClusterOrganizer(static_data, is_online=False, train=True, refit=False)
    cluster_organizer.fit()
    cluster_organizer.create_clusters_and_cvs()


def feature_selection(static_data):
    feature_selector = FeatureSelector(static_data, recreate=RECREATE_DATASETS, online=False, train=True)
    feature_selector.fit(recreate_lstm=False)
    feature_selector.create_backup()


if __name__ == '__main__':
    module = importlib.import_module(f'configuration.config')
    config = getattr(module, 'config')
    static_data = initializer(config())
    dataset = DatasetCreator(static_data, recreate=RECREATE_DATASETS, train=True)
    # create_datasets(dataset)
    # preprocess_data(static_data)
    clustering_fit(static_data)
    # feature_selection(static_data)

