# -*- coding: utf-8 -*-

__all__ = ["period_confirm"]

import numpy as np
import pandas as pd


def period_confirm(series: pd.Series, cfg: dict):
    """
    detect and confirm the period
    :param series: pd.Series
    :param cfg: configuration dict
    :return:
    """
    flag = period_confirm_stl_variance(series, cfg)
    if flag:
        cfg['has_period'] = flag


def period_confirm_stl_variance(series, cfg):
    """
    :param series:  pd.Series
    :param period_candidate:
    :param cfg:
    :return:
    """

    tmp_std = compute_variance_min_max(series, cfg['period'])
    if tmp_std < 0.15:
        return True
    else:
        return False


def compute_variance_min_max(array, period):
    """
    This function is used to compute normalized diff variance. First, we normalize the data within a period using
    MinMaxScaler; Second, we compute the first order difference between consecutive period data; Finally, we compute
    the diff variance
    :param array: np.ndarray. The data
    :param period: The detected period by our period detection module
    :return:
    variance
    """
    list = []
    num_period = int(np.ceil(len(array) / period))
    for i in range(num_period):
        start = i * period
        end = (i + 1) * period
        if end >= len(array):
            end = len(array)

        tmp_array = array[start:end]
        tmp_min = float(np.min(tmp_array))
        tmp_max = float(np.max(tmp_array))
        for j in tmp_array:
            if (tmp_max - tmp_min) == 0:
                list.append(0.0)
            else:
                list.append((float(j) - tmp_min + 0.0) / (tmp_max - tmp_min))

    retrun_list = []
    for c in range(period, len(list)):
        retrun_list.append(list[c] - list[c - period])

    return_array = np.array(retrun_list)
    std_value = np.std(return_array)

    return std_value
