import sys
import os
import joblib
import copy

print(os.getcwd())
sys.path.append(os.getcwd())


from eforecast.init.initialize import initializer
from configuration.config import config
static_data = initializer(config())
from eforecast.prediction.predict import Predictor
from eforecast.prediction.evaluate import Evaluator
from eforecast.combine_predictions.combine_predictions_fit import CombinerFit
# predictor = Predictor(static_data, train=True)
# predictor.predict_regressors(average=True, parallel=False)
# # combiner = CombinerFit(static_data, refit=True)
# # combiner.fit_methods()
# predictor.predict_combine_methods()
# predictor.compute_predictions_averages(only_methods=False, only_combine_methods=False)
evaluator = Evaluator(static_data, refit=True)
evaluator.evaluate_averages()
