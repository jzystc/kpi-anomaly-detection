# -*- coding: utf-8 -*-

__all__ = [
    "discrete_confirm",
    "denoise_train_data",
]

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess


def discrete_confirm(series, cfg) -> dict:
    """
    get the profile of the metric, such as stableness etc.
    :param seriesDF:  pd.DataFrame
    :param cfg:  configuration dict
    :return:
    """
    discrete = _discrete_ratio_of(series)
    cfg["discrete"] = discrete
    return discrete


def _discrete_ratio_of(series, judge_number=20, max_ratio=0.9) -> bool:
    """
    TODO 再添加一个过过滤规则, 就是如果某个值占比达到95%以上，那么也直接判定为离散指标
    :param series:  pd.Series
    :return:
    """
    not_nan_series = list()
    for item in series:
        if not pd.isnull(item):
            not_nan_series.append(item)

    unique_elements, counts_elements = np.unique(not_nan_series, return_counts=True)
    max_value_ratio = max(counts_elements) / len(not_nan_series)

    if (len(unique_elements) < judge_number) or (max_value_ratio > max_ratio):
        return True
    else:
        return False


def denoise_train_data(series: pd.Series, windowsize=10) -> pd.Series:
    '''remove point-anomalies through loess smoothing
    '''
    smooth_data = lowess(series.values, range(len(series)), frac=(windowsize) / (2 * len(series)))
    smoothed_series = pd.Series(data=smooth_data[:, 1], index=series.index)
    error_series = (series - smoothed_series).abs()

    upper_bound = error_series.mean() + 3 * error_series.std()
    filtered_series = series.copy()
    filtered_series[error_series > upper_bound] = np.nan
    filtered_series.iloc[0] = series.iloc[0]
    filtered_series.iloc[-1] = series.iloc[-1]

    return filtered_series.interpolate()
