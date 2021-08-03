from math import fabs
import os
import pickle
import warnings
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from statsmodels.tsa.ar_model import AR_DEPRECATION_WARN
from kpi.processor.series_type import type_tag
from kpi.detector.predictors import (
    LGBMPredictor,
    XGBPredictor,
    GaussianMixtureDetector,
    BoxPlotDetector,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import optuna
from optuna.samplers import TPESampler
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction.feature_calculators import *
from pyod.models.iforest import IForest


def feature_engineering(data, window_size=5, norm=True, min_max_norm=False):
    kpi_series = data["value"].astype(float)
    # 填充缺失值
    kpi_series = kpi_series.interpolate()
    if norm:
        if min_max_norm:
            kpi_series = (kpi_series - kpi_series.min()) / (
                kpi_series.max() - kpi_series.min()
            )  # 即简单实现标准化
        else:
            kpi_series = (kpi_series - kpi_series.mean()) / (kpi_series.std())
    mean = data["value"].mean()
    median = data["value"].median()
    var = data["value"].var()  # 方差
    # mode = data['value'].mode()[0]  # 众数
    max_ = data["value"].max()
    min_ = data["value"].min()
    std = data["value"].std()  # 标准差
    skew = data["value"].skew()  # 偏度
    try:
        cv = std / (mean)  # 变异度
    except Exception:
        cv = 0

    rolling_mean = data["value"].rolling(window_size).mean()
    rolling_max = data["value"].rolling(window_size).max()
    rolling_min = data["value"].rolling(window_size).min()
    rolling_var = data["value"].rolling(window_size).var()
    rolling_skew = data["value"].rolling(window_size).skew()
    rolling_std = data["value"].rolling(window_size).std()
    rolling_median = data["value"].rolling(window_size).median()
    try:
        rolling_cv = rolling_std / (rolling_mean)
    except Exception:
        rolling_cv = 0
    # data[['mean', 'var', 'mode', 'skew', 'std', 'max', 'min', 'median', 'cv']] = pd.DataFrame(
    #     [[mean, var, mode, skew, std, max_, min_, median, cv]], index=data.index)
    data.insert(data.shape[1], "mean", mean)
    data.insert(data.shape[1], "var", var)
    # data.insert(data.shape[1], 'mode', mode)
    data.insert(data.shape[1], "skew", skew)
    data.insert(data.shape[1], "max", max_)
    data.insert(data.shape[1], "min", min_)
    data.insert(data.shape[1], "median", median)
    data.insert(data.shape[1], "std", std)
    data.insert(data.shape[1], "cv", cv)
    data.insert(data.shape[1], "rolling_mean", rolling_mean)
    data.insert(data.shape[1], "rolling_var", rolling_var)
    data.insert(data.shape[1], "rolling_skew", rolling_skew)
    data.insert(data.shape[1], "rolling_max", rolling_max)
    data.insert(data.shape[1], "rolling_min", rolling_min)
    data.insert(data.shape[1], "rolling_median", rolling_median)
    data.insert(data.shape[1], "rolling_std", rolling_std)
    data.insert(data.shape[1], "rolling_cv", rolling_cv)

    ts = pd.Series(data["value"])  # 数据x假设已经获取
    feat1 = abs_energy(ts)
    feat2 = absolute_sum_of_changes(ts)
    param = [{"coeff": 0, "k": 10}]
    feat3 = ar_coefficient(ts, param)[0][1]
    feat4 = autocorrelation(ts, 2)
    param = [{"attr": "pvalue"}]
    feat5 = augmented_dickey_fuller(ts, param)[0][1]
    feat6 = approximate_entropy(ts, 10, 0.1)
    feat7 = kurtosis(ts)
    diff_1 = ts.diff(1)
    # diff_1.fillna(diff_1.mean(), inplace=True)
    diff_24 = ts.diff(24)
    # diff_24.fillna(diff_24.mean(), inplace=True)
    data.insert(data.shape[1], "diff_24", diff_24)
    data.insert(data.shape[1], "diff_1", diff_1)
    data.insert(data.shape[1], "abs_energy", feat1)
    data.insert(data.shape[1], "absolute_sum_of_changes", feat2)
    data.insert(data.shape[1], "ar_coefficient", feat3)
    data.insert(data.shape[1], "autocorrelation", feat4)

    data.insert(data.shape[1], "augmented_dickey_fuller", feat5)

    data.insert(data.shape[1], "approximate_entropy", feat6)
    data.insert(data.shape[1], "kurtosis", feat7)
    for column in list(data.columns[data.isnull().sum() > 0]):
        mean_val = data[column].mean()
        data[column].fillna(mean_val, inplace=True)
    return data


def create_model(trial):
    num_leaves = trial.suggest_int("num_leaves", 2, 31)
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 8)
    min_child_samples = trial.suggest_int("min_child_samples", 100, 1200)
    learning_rate = trial.suggest_uniform("learning_rate", 0.0001, 0.99)
    min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 5, 90)
    bagging_fraction = trial.suggest_uniform("bagging_fraction", 0.0001, 1.0)
    feature_fraction = trial.suggest_uniform("feature_fraction", 0.0001, 1.0)
    model = LGBMClassifier(
        num_leaves=num_leaves,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        min_data_in_leaf=min_data_in_leaf,
        learning_rate=learning_rate,
        feature_fraction=feature_fraction,
        random_state=666,
    )
    return model


features = [
    "value",
    "mean",
    "var",
    # 'mode',
    "skew",
    "std",
    "max",
    "min",
    "median",
    "cv",
    "rolling_mean",
    "rolling_var",
    "rolling_skew",
    "rolling_std",
    "rolling_max",
    "rolling_min",
    "rolling_median",
    "rolling_cv",
    "abs_energy",
    "absolute_sum_of_changes",
    "ar_coefficient",
    "autocorrelation",
    "augmented_dickey_fuller",
    "approximate_entropy",
    "kurtosis",
    "diff_1",
    "diff_24",
]

target = "label"


def extract_features_by_tsfresh(df):
    features = extract_features(
        df, column_id="kpi_id", column_sort="start_time", column_value="value"
    )
    features.index.names = ["kpi_id"]
    features = features.dropna(axis=1)
    feature_names = features.columns.values
    # print(feature_names)
    df = pd.merge(df, features, how="outer", on="kpi_id")
    return df, feature_names


def save_model(model, model_name):
    if ".pkl" not in model_name:
        model_name = model_name + ".pkl"
    if not os.path.exists("./model"):
        os.makedirs("./model")
    model_path = os.path.join("./model/", model_name)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


def objective(trial, train_df, test_df):
    model = create_model(trial)
    model.fit(train_df[features], train_df[target])
    score = f1_score(test_df[target].values, model.predict(test_df[features]))
    return score


def objective(trial, train_x, train_y, test_x, test_y):
    model = create_model(trial)
    model.fit(train_x[features], train_y)
    score = f1_score(test_y.values, model.predict(test_x[features]))
    return score


def get_cfg(sensitivity=3, period_day_hours=24, one_day_sample_nums=60):
    cfg = {
        "period": int(period_day_hours * 60 / one_day_sample_nums),
        "sensitivity": sensitivity,
    }
    return cfg


def train(kpi_df, tuning=False, n_trials=20, norm=True):
    params = {
        "bagging_fraction": 0.5817242323514327,
        "feature_fraction": 0.6884588361650144,
        "learning_rate": 0.42887924851375825,
        "max_depth": 6,
        "min_child_samples": 946,
        "min_data_in_leaf": 47,
        "n_estimators": 169,
        "num_leaves": 29,
        "random_state": 666,
        # 'is_unbalance':False,
        "class_weight": "balanced",
    }
    sampler = TPESampler(seed=666)
    # kpi_df = get_data_reference(dataset=dataset, dataset_entity=dataset_entity).to_pandas_dataframe()
    # kpis = kpi_df['KPI ID'].value_counts()
    scores = []
    # 获取不同kpi id的时间序列
    groups = kpi_df.groupby("kpi_id")
    for kpi_id, df in groups:
        kpi_series = df["value"]
        # cfg = get_cfg()
        # kpi_series = type_tag(kpi_series, cfg)
        df["value"] = kpi_series
        labels = df["label"]
        # print(pd.unique(labels))

        df = df[["start_time", "value", "kpi_id"]]
        df = feature_engineering(df, window_size=5, norm=norm)
        # df, feature_names = extract_features_by_tsfresh(df)
        # train_x, test_x, train_y, test_y = train_test_split(
        #     df, labels, shuffle=True, test_size=0.2, random_state=666)
        train_x, test_x, train_y, test_y = df, df, labels, labels

        # try:
        #     train_x, test_x, train_y, test_y = train_test_split(
        #         df, labels, shuffle=True, test_size=0.2, stratify=labels,random_state=666)
        # except ValueError:
        #     train_x, test_x, train_y, test_y = train_test_split(
        #         df, labels, shuffle=True, test_size=0.2, random_state=666)
        # features = feature_names
        # train_x = feature_engineering(train_x, window_size=5, norm=norm)
        # test_x = feature_engineering(test_x, window_size=5, norm=norm)
        if tuning:
            # 超参调优
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(
                lambda trial: objective(trial, train_x, train_y, test_x, test_y),
                n_trials=n_trials,
            )
            params = study.best_params

        kpi_predictor = IForest(
            n_estimators=200,
            max_samples="auto",
            contamination=0.2,
            max_features=0.7,
            bootstrap=False,
            n_jobs=4,
            behaviour="old",
            random_state=None,
            verbose=0,
        )
        # results= cross_val(kpi_predictor,df[features],labels,scoring="f1",cv=5,return_estimator=True)
        # results= cross_val_score(kpi_predictor,df[features],labels,scoring="f1",cv=5)
        # print(results)
        kpi_predictor.fit(train_x[features], train_y)
        # kpi_predictor.fit(df[features], labels)
        score = f1_score(
            test_y.values,
            kpi_predictor.predict(test_x[features]),
            average="binary",
            zero_division=0,
        )
        print(kpi_id, "f1:%.2f" % score)
        scores.append(score)
        # save_model(model=kpi_predictor, model_name=kpi_id)
    print("average f1 score:", np.average(scores))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    data_path = "./data/KPI_Train.csv"
    df = pd.read_csv(data_path)
    train(df, tuning=False, n_trials=20, norm=False)
