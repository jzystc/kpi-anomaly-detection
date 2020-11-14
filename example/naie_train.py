import os
import pickle
from naie.context import Context
from kpi.processor.series_type import type_tag
from kpi.detector.predictors import XGBPredictor, GaussianMixtureDetector, BoxPlotDetector
from naie.datasets import get_data_reference

def train(dataset, dataset_entity):
    kpi_df = get_data_reference(dataset=dataset, dataset_entity=dataset_entity).to_pandas_dataframe()
    kpis = kpi_df['kpi_id'].value_counts()
    #获取不同kpi id的时间序列
    for i in range(0, len(kpis)):
        kpi_id = kpis.index[i]
        df = kpi_df.loc[kpi_df['kpi_id'] == kpi_id]
        kpi_series = df['value'].astype(float)
        kpis_data = kpi_series.interpolate()
        train_data = kpis_data.head(len(kpis_data) - 24 * 5)
        # config parameter
        sensitivity = 3
        period_day_hours = 24
        # 采样频率
        one_day_sample_nums = 60
        cfg = {
            'period': int(period_day_hours * 60 / one_day_sample_nums),
            'sensitivity': sensitivity
        }
        # kpi type tag 给指标打tag, 有周期，还是无周期
        train_data = type_tag(train_data, cfg)

        # train model
        if cfg.get('has_period') is not None:
            KpiPredictor = XGBPredictor(cfg=cfg)
        else:
            KpiPredictor = GaussianMixtureDetector(cfg=cfg)
        KpiPredictor.train_model(series=train_data)
        model_name = kpi_id + '.pkl'

        # save model
        model_path = os.path.join(Context.get_output_path(), "model", model_name)
        with open(model_path, 'wb') as f:
            pickle.dump(KpiPredictor, f)

if __name__ == "__main__":
    train("data", "train")

