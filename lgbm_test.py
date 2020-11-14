import pandas as pd
import pickle
import os

from lgbm_train import feature_engineering

features = [
    'value', 'mean', 'var', 'mode', 'skew', 'std', 'max', 'min', 'median', 'cv', 'rolling_mean', 'rolling_var',
    'rolling_skew', 'rolling_std', 'rolling_max', 'rolling_min', 'rolling_median', 'rolling_cv'
]

target = 'label'


def detect(kpi_df):
    # kpi_df = get_data_reference(dataset=dataset, dataset_entity=dataset_entity).to_pandas_dataframe()
    # kpis = kpi_df['kpi_id'].value_counts()
    res_list = []
    kpis_list = ['02e99bd4f6cfb33f' '9bd90500bfd11edb' 'da403e4e3f87c9e0'
                 'a5bf5d65261d859a' '18fbb1d5a5dc099d' '09513ae3e75778a3'
                 'c58bfcbacb2822d1' '1c35dbf57f55f5e4' '046ec29ddf80d62e'
                 '07927a9a18fa19ae' '54e8a140f6237526' 'b3b2e6d1a791d63a'
                 '8a20c229e9860d0c' '769894baefea4e9e' '76f4550c43334374'
                 'e0770391decc44ce' '8c892e5525f3e491' '40e25005ff8992bd'
                 'cff6d3c01e6a6bfa' '71595dd7171f4540' '7c189dd36f048a6c'
                 'a40b1df87e3f1c87' '8bef9af9a922e0b3' 'affb01ca2b4f0b45'
                 '9ee5879409dccef9' '88cf3a776ba00e7c']

    for kpi_id in kpis_list:
        # kpi_id = kpis.index[i]
        df = kpi_df.loc[kpi_df['kpi_id'] == kpi_id]
        kpi_series = df['value'].astype(float)
        kpis_data = kpi_series.interpolate()
        df['value'] = kpis_data
        df = feature_engineering(df)
        model_path = os.path.join(
            Context.get_output_path(), "model",  kpi_id+'.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            model.predict(df[features])
            label = model.predict(df[features])
            res = {
                'start time': list(df.iloc[:, 0]),
                'end time': list(df.iloc[:, 1]),
                'kpi_id': kpi_id,
                'label': label
            }
        resDF = pd.DataFrame(res)
        res_list.append(resDF)
    res_all = pd.concat(res_list)
    out_path = os.path.join(Context.get_output_path(), "result" + '.csv')
    with open(out_path, 'w') as f:
        res_all.to_csv(f, index=False)


if __name__ == "__main__":
    detect("data", "test")
