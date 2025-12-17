import numpy as np
import pandas as pd


def transform_rated(rated, y):
    if len(y.shape) > 1:
        if y.shape[1] > 1:
            rated = y.values if rated is None else rated
        else:
            rated = y.values.ravel() if rated is None else rated
    else:
        rated = y.values if rated is None else rated
    return rated


def flat_target(targ):
    if len(targ.shape) == 2:
        if targ.shape[1] == 1:
            targ = targ.ravel()
    return targ


def compute_metrics(pred, y, rated, predictor_name, multi_output=False):
    rated = transform_rated(rated, y)
    y_np = flat_target(y.values)
    pred_np = flat_target(pred.values)
    w = np.array([0.1, 0.1, 0.1, 0.5, 0.2])
    err1 = np.square(pred_np - y_np)
    err2 = np.square(y_np - ((w * y_np) / w.sum()).mean(axis=1, keepdims=True))

    r2 = np.nanmean(1 - (w*err1).sum(axis=1) / (w*err2).sum(axis=1))
    sse = np.nanmean((w*err1).sum(axis=1))
    tsse = np.nanmean((w*err2).sum(axis=1))
    res = pd.DataFrame([[r2, sse, tsse]], columns=['r2', 'sse', 'tsse'],
                       index=[predictor_name])
    res['average'] = res.mean(axis=1)
    return res
