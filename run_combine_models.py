#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module handles the combination of different forecasting models and their predictions.
It provides functionality for model evaluation, prediction combination, and ensemble creation.
"""

import sys
import os
import importlib
import pandas as pd
print(os.getcwd())
sys.path.append(os.getcwd())
from eforecast.init.initialize import initializer

from eforecast.common_utils.devops_utils import get_results
from eforecast.combine_predictions.combine_predictions_fit import CombinerFit
from eforecast.prediction.predict import Predictor
from eforecast.prediction.evaluate import Evaluator

RECREATE_DATASETS = False


def predict_methods(static_data, train=True, average=False, parallel=False):
    """
    Make predictions using individual forecasting methods.
    
    Args:
        static_data: Configuration and static data for the model
        train (bool): Whether to use training data
        resampling (bool): Whether to use resampled data
        eval_tests (bool): Whether to evaluate test cases
    """
    predictor = Predictor(static_data, train=train)
    predictor.predict_regressors(average=average, parallel=parallel)

def evaluate_methods(static_data):
    """
    Evaluate the performance of individual forecasting methods.
    
    Args:
        static_data: Configuration and static data for the model
    """
    evaluator = Evaluator(static_data, refit=True)
    evaluator.evaluate_methods()

def evaluate_averages(static_data):
    """
    Evaluate the performance of averaged predictions.
    
    Args:
        static_data: Configuration and static data for the model
    """
    evaluator = Evaluator(static_data, refit=True)
    evaluator.evaluate_averages()

def combine_methods(static_data):
    """
    Combine different forecasting methods using the CombinerFit class.
    
    Args:
        static_data: Configuration and static data for the model
    """
    combiner = CombinerFit(static_data, refit=True)
    combiner.fit_methods()

def predict_combine_methods(static_data, train=True):
    """
    Make predictions using combined forecasting methods.
    
    Args:
        static_data: Configuration and static data for the model
        train (bool): Whether to use training data
        resampling (bool): Whether to use resampled data
    """
    predictor = Predictor(static_data, train=train)
    predictor.predict_combine_methods()

def compute_averages_methods(static_data, train=True, only_methods=False, only_combine_methods=False):
    """
    Compute average predictions from different methods.
    
    Args:
        static_data: Configuration and static data for the model
        train (bool): Whether to use training data
        resampling (bool): Whether to use resampled data
        only_methods (bool): Whether to use only individual methods
        only_combine_methods (bool): Whether to use only combined methods
    """
    predictor = Predictor(static_data, train=train)
    predictor.compute_predictions_averages(only_methods=only_methods, only_combine_methods=only_combine_methods)

def combine_models(static_data, combine_methods=None):
    """
    Combine different models and fit concatenated neural networks.
    
    Args:
        static_data: Configuration and static data for the model
        combine_methods (list): List of methods to combine
    """
    combiner = CombinerFit(static_data, refit=True)
    if combine_methods is not None:
        combiner.fit_models(combine_methods)

def predict_combine_models(static_data, train=True, combine_methods=None):
    """
    Make predictions using combined models and concatenated neural networks.
    
    Args:
        static_data: Configuration and static data for the model
        train (bool): Whether to use training data
        resampling (bool): Whether to use resampled data
        combine_methods (list): List of methods to combine
    """
    predictor = Predictor(static_data, train=train)
    predictor.predict_combine_models(combine_methods)

def evaluate_models(static_data):
    """
    Evaluate the performance of combined models.
    
    Args:
        static_data: Configuration and static data for the model
    """
    evaluator = Evaluator(static_data, refit=True)
    evaluator.evaluate_models()

def step1(static_data, parallel=False):
    """
    First step in the model combination pipeline:
    - Predict using individual methods
    - Evaluate models
    
    Args:
        static_data: Configuration and static data for the model
    """
    # predict_methods(static_data, train=True, average=False, parallel=parallel)
    predict_methods(static_data, train=False, average=False, parallel=parallel)
    evaluate_methods(static_data)

def step2(static_data, parallel=False):
    """
    Second step in the model combination pipeline:
    - Predict using individual and combined methods
    - Compute averages
    - Evaluate results
    
    Args:
        static_data: Configuration and static data for the model
    """
    predict_methods(static_data, train=True, average=True, parallel=parallel)
    predict_methods(static_data, train=False, average=True, parallel=parallel)
    combine_methods(static_data)
    predict_combine_methods(static_data, train=True)
    compute_averages_methods(static_data, train=True, only_methods=False, only_combine_methods=False)
    predict_combine_methods(static_data, train=False)
    compute_averages_methods(static_data, train=False, only_methods=False, only_combine_methods=False)
    evaluate_averages(static_data)

def step3(static_data):
    """
    Third step in the model combination pipeline:
    - Combine models using kmeans
    - Make predictions
    - Evaluate results
    
    Args:
        static_data: Configuration and static data for the model
    """
    combine_models(static_data, combine_methods=['kmeans'])
    predict_combine_models(static_data, train=True, combine_methods=['kmeans'])
    predict_combine_models(static_data, train=False, combine_methods=['kmeans'])
    evaluate_models(static_data)

def run_step1(project_name_list, parallel=False):
    name = '.'.join(project_name_list)
    module = importlib.import_module(f'{name}.configuration.config')
    config = getattr(module, 'config')
    static_data = initializer(config())
    step1(static_data, parallel=False)

def run_step2(project_name_list, parallel=False):
    name = '.'.join(project_name_list)
    module = importlib.import_module(f'{name}.configuration.config')
    config = getattr(module, 'config')
    static_data = initializer(config())
    step2(static_data, parallel=False)

def run_step3(project_name_list):
    name = '.'.join(project_name_list)
    module = importlib.import_module(f'{name}.configuration.config')
    config = getattr(module, 'config')
    static_data = initializer(config())
    step3(static_data)

def partial_results(static_data):
    import joblib
    clusters = joblib.load(os.path.join(static_data['path_model'], 'clusters.pickle'))
    best = get_results(clusters, 'TIMM', 0)

if __name__ == '__main__':
    # Initialize configuration and run the model combination pipeline
    from configuration.config import config
    static_data = initializer(config())
    # partial_results(static_data)
    step1(static_data, parallel=False)
    # step2(static_data, parallel=False)
    # step3(static_data)
