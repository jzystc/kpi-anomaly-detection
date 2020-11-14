'''
give tag for different time series.
'''

__all__ = ["type_tag"]

import pandas as pd
from .period_confirm import period_confirm
from kpi.processor.preprocessing import denoise_train_data, discrete_confirm


def type_tag(series: pd.Series, cfg: dict):
    '''  give tag for kpi series, discrete, period, non-period
        "has_period" key will be added in cfg dict.
        "period" will be added if "has_period" is True
    '''
    ## series preprocess
    denoised_series = denoise_train_data(series)

    if discrete_confirm(series, cfg):
        return denoised_series
    # 在period_confirm中确定了最后的周期数 key为"period", 并加入到cfg中
    period_confirm(denoised_series, cfg)
    return denoised_series
