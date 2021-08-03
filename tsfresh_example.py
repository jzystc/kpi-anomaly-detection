import pandas as pd
import tsfresh as tsf
from tsfresh import extract_features
from tsfresh.feature_extraction.feature_calculators import *
from tsfresh.feature_extraction.settings import *


class MyFCParmameters(MinimalFCParameters):
    def __init__(self, params: dict=None):
        super(MyFCParmameters, self).__init__()
        if params!=None:
            for k, v in params.items():
                self[k] = v
        


if __name__ == '__main__':
    # download_robot_execution_failures()
    # timeseries, y = load_robot_execution_failures()
    data_path = './data/KPI_Train.csv'
    df = pd.read_csv(data_path)
    y = df[['label']]
    # print(y.head(10))
    df = df[['start_time', 'value', 'kpi_id']]
    groups=df.groupby('kpi_id')
    for kpi_id,kpi_df in groups:
        ts = pd.Series(kpi_df['value'])  #数据x假设已经获取
        feat1=abs_energy(ts)
        feat2=absolute_sum_of_changes(ts)
        param = [{'coeff': 0, 'k': 10}]
        feat3=ar_coefficient(ts, param)[0][1]
        feat4=autocorrelation(ts, 2)
        # print(feat1,feat2,feat3,feat4)
        param = [{'attr': 'pvalue'}]
        feat5=augmented_dickey_fuller(ts, param)[0][1]
        feat6=approximate_entropy(ts, 10, 0.1)
        print(feat6)


    # print(len(pd.unique(kpi_series['kpi_id'])))
    # print(kpi_series.head(10))
    # print(len(y),len(kpi_series))
    # params = MyFCParmameters({''})
    # features = extract_features(kpi_series, default_fc_parameters=params,
    #                             column_id="kpi_id", column_sort="start_time", column_value="value")
    # print(features.info())
    # features.index.names=['kpi_id']
    # print(features.head(5))
    # features = features.dropna(axis=1)
    # print(features.info())
    # df=pd.merge(kpi_series,features,how="outer",on="kpi_id")
    # print(df.shape)
    # print(extracted_features.shape)
    # print(extracted_features.head(10))
    # impute(extracted_features)
    # features_filtered = select_features(extracted_features, y)
    # print(features_filtered.shape)
