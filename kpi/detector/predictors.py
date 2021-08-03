# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xgboost
from lightgbm import LGBMClassifier
from sklearn.mixture import GaussianMixture


class LGBMPredictor(LGBMClassifier):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)
        self.detector = 'period'


class XGBPredictor:
    def __init__(self, cfg: dict):
        self.period = cfg['period']
        self.model = None
        self.features = None  # 预测模型需要的特征(使用历史一个周期的数据做特征滑动)
        self.errs = None  #
        self.detector = 'period'
        self.sens = cfg['sensitivity']

    def train_model(self, series: pd.Series):
        """
        train the XGB regressor model
        :return:
        """
        # 历史模型训练
        samples = np.array([series.values[i - self.period - 1: i]
                            for i in range(self.period + 1, len(series))])
        Xs = samples[:, :-1]
        Ys = samples[:, -1:]
        self.features = series[-self.period:]
        self.model = xgboost.XGBRegressor(
            n_estimators=100, max_depth=5, random_state=0)
        self.model.fit(Xs, Ys)
        self.errs = self.predict(list(series))

    # 基于历史特征训练模型
    def predict(self, features: list):
        err = []
        for i in range(len(features) - self.period):
            tmp_feature = features[i:(i + self.period)]
            tmp_res = self.model.predict(np.array(tmp_feature).reshape(1, -1))
            err.append(abs(tmp_res[0] - features[i + self.period]))
        return err

    def detect(self, series_new: pd.Series):
        err_len = len(self.errs)
        series_predict_res = self.predict(series_new)
        num_new = len(series_predict_res)
        res_flag = []
        for i in range(num_new):
            errs = self.errs
            errs.append(series_predict_res[i])
            lower_bound = np.array(errs).mean() - \
                self.sens * np.array(errs).std()
            up_bound = np.array(errs).mean() + self.sens * np.array(errs).std()
            if series_predict_res[i] < lower_bound or series_predict_res[i] > up_bound:
                tmp_flag = 1
            else:
                tmp_flag = 0
            res_flag.append(tmp_flag)
            self.errs = errs[-err_len:]

        return res_flag


class GaussianMixtureDetector:
    def __init__(self, cfg: dict = None):
        self.model = GaussianMixture(n_components=3)
        self.sens = cfg['sensitivity']
        self.lower_bound = None
        self.detector = 'nonperiod'

    def train_model(self, series: pd.Series):
        self.model.fit(series.values.reshape(-1, 1))
        scores = self.model.score_samples(series.values.reshape(-1, 1))
        lower_bound = scores.mean() - self.sens * scores.std()
        self.lower_bound = lower_bound

    def predict(self, series_new: pd.Series):
        scores_new = self.model.score_samples(series_new.values.reshape(-1, 1))
        return [(score_new, self.lower_bound) for score_new in scores_new]

    def detect(self, series_new: pd.Series):
        num_new = len(series_new)
        series_predict_res = self.predict(series_new=series_new)
        res_flag = []
        for i in range(num_new):
            lower_bound = series_predict_res[i][1]
            if series_predict_res[i][0] <= lower_bound:
                tmp_flag = 1
            else:
                tmp_flag = 0
            res_flag.append(tmp_flag)

        return res_flag


class BoxPlotDetector:
    def __init__(self, cfg: dict):
        self.upbound = None  #
        self.lowbound = None  #
        self.sens = cfg['sensitivity']
        self.detector = 'discrete'

    def train_model(self, series: pd.Series):
        """
        train the XGB regressor model
        :return:
        """
        quntile_1 = 25
        quntile_3 = 75
        q1 = np.percentile(series, quntile_1)
        q3 = np.percentile(series, quntile_3)
        delta = (q3 - q1) * 0.5
        iqr = delta * self.sens
        ucl = q3 + iqr
        lcl = q1 - iqr
        self.lowbound = lcl
        self.upbound = ucl

    def detect(self, series_new: pd.Series):
        num_new = len(series_new)
        series_new = series_new.values
        res_flag = []
        for i in range(num_new):
            lower_bound = self.lowbound
            upper_bound = self.upbound
            if series_new[i] < lower_bound or series_new[i] > upper_bound:
                tmp_flag = 1
            else:
                tmp_flag = 0
            res_flag.append(tmp_flag)
        return res_flag
