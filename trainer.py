import os
import pickle
from kpi.processor.series_type import type_tag
from kpi.detector.predictors import LGBMPredictor, XGBPredictor, GaussianMixtureDetector, BoxPlotDetector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import optuna
from optuna.samplers import TPESampler


def feature_engineering(data, window_size=5):
    mean = data['value'].mean()
    median = data['value'].median()
    var = data['value'].var()  # 方差
    mode = data['value'].mode()[0]  # 众数
    max_ = data['value'].max()
    min_ = data['value'].min()
    std = data['value'].std()  # 标准差
    skew = data['value'].skew()  # 偏度
    if mean == 0:
        cv = 0
    else:
        cv = std / mean  # 变异度
    rolling_mean = data['value'].rolling(window_size).mean()
    rolling_max = data['value'].rolling(window_size).max()
    rolling_min = data['value'].rolling(window_size).min()
    rolling_var = data['value'].rolling(window_size).var()
    rolling_skew = data['value'].rolling(window_size).skew()
    rolling_std = data['value'].rolling(window_size).std()
    rolling_median = data['value'].rolling(window_size).median()
    rolling_cv = rolling_std / rolling_mean
    data[['mean', 'var', 'mode', 'skew', 'std', 'max', 'min', 'median', 'cv']] = pd.DataFrame(
        [[mean, var, mode, skew, std, max_, min_, median, cv]], index=data.index)
    data['rolling_mean'], data['rolling_var'], data['rolling_skew'], data['rolling_max'], data['rolling_min'], data[
        'rolling_median'], data['rolling_std'], data[
        'rolling_cv'] = rolling_mean, rolling_var, rolling_skew, rolling_max, rolling_min, rolling_median, rolling_std, rolling_cv
    return data


def create_model(trial):
    num_leaves = trial.suggest_int("num_leaves", 2, 31)
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 8)
    min_child_samples = trial.suggest_int('min_child_samples', 100, 1200)
    learning_rate = trial.suggest_uniform('learning_rate', 0.0001, 0.99)
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 5, 90)
    bagging_fraction = trial.suggest_uniform('bagging_fraction', 0.0001, 1.0)
    feature_fraction = trial.suggest_uniform('feature_fraction', 0.0001, 1.0)
    model = LGBMClassifier(
        num_leaves=num_leaves,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        min_data_in_leaf=min_data_in_leaf,
        learning_rate=learning_rate,
        feature_fraction=feature_fraction,
        random_state=666
    )
    return model


features = [
    'value', 'mean', 'var', 'mode', 'skew', 'std', 'max', 'min', 'median', 'cv', 'rolling_mean', 'rolling_var',
    'rolling_skew', 'rolling_std', 'rolling_max', 'rolling_min', 'rolling_median', 'rolling_cv'
]

target = 'label'


def objective(trial, train_df, test_df):
    model = create_model(trial)
    model.fit(train_df[features], train_df[target])
    score = f1_score(test_df[target].values, model.predict(test_df[features]))
    return score


def train(kpi_df):
    params = {
        'bagging_fraction': 0.5817242323514327,
        'feature_fraction': 0.6884588361650144,
        'learning_rate': 0.42887924851375825,
        'max_depth': 6,
        'min_child_samples': 946,
        'min_data_in_leaf': 47,
        'n_estimators': 169,
        'num_leaves': 29,
        'random_state': 666,
        'is_unbalance':True
    }
    sampler = TPESampler(seed=666)
    # kpi_df = get_data_reference(dataset=dataset, dataset_entity=dataset_entity).to_pandas_dataframe()
    # kpis = kpi_df['KPI ID'].value_counts()
    scores = []
    # 获取不同kpi id的时间序列
    groups = kpi_df.groupby('kpi_id')
    for kpi_id, df in groups:
        # kpi_id = kpis.index[i]
        # df = kpi_df.loc[kpi_df['KPI ID'] == kpi_id]
        kpi_series = df['value'].astype(float)
        # 填充缺失值
        kpis_data = kpi_series.interpolate()

        # config parameter
        sensitivity = 3
        period_day_hours = 24
        # 采样频率
        one_day_sample_nums = 60
        cfg = {
            'period': int(period_day_hours * 60 / one_day_sample_nums),
            'sensitivity': sensitivity
        }
        # cfg
        # kpi type tag 给指标打tag, 有周期，还是无周期
        kpis_data = type_tag(kpis_data, cfg)
        df['value'] = kpis_data
        df = feature_engineering(df, window_size=5)
        train_df, test_df = train_test_split(
            df, shuffle=False, test_size=0.2)
        # train model

        if cfg.get('has_period') is not None:
            #超参调优
            # study = optuna.create_study(direction="maximize", sampler=sampler)
            # study.optimize(lambda trial: objective(
            #     trial, train_df, test_df), n_trials=10)
            # params = study.best_params
            kpi_predictor = LGBMPredictor(**params)
            kpi_predictor.fit(train_df[features], train_df[target])
            # KpiPredictor.fit(train_df[features], train_df[target])
            score = f1_score(test_df[target].values,
                             kpi_predictor.predict(test_df[features]))
            print('LGBMPredictor', kpi_id, 'f1 score: ', score)
            scores.append(score)
        # elif cfg.get("discrete") is not None:
        #     kpi_predictor = BoxPlotDetector(cfg=cfg)
        #     kpi_predictor.train_model(series=train_df['value'])
        #     score = f1_score(test_df[target].values,
        #                      kpi_predictor.detect(series_new=test_df['value']))
        #     print('BoxPlotDetector', kpi_id, 'f1 score: ', score)
        #     scores.append(score)

        # elif cfg.get("nonperiod") is not None:
        else:
            kpi_predictor = GaussianMixtureDetector(cfg=cfg)
            kpi_predictor.train_model(series=train_df['value'])
            score = f1_score(test_df[target].values,
                             kpi_predictor.detect(series_new=test_df['value']))
            print('GaussianMixtureDetector', kpi_id, 'f1 score: ', score)
            scores.append(score)
    print('average f1 score:', np.average(scores))


if __name__ == "__main__":
    data_path = './data/KPI_Train.csv'
    df = pd.read_csv(data_path)
    # print(df.head)
    train(df)
