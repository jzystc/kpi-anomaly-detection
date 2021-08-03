from numpy.linalg.linalg import norm
import pandas as pd
import pickle
import os

from lgbm_train import feature_engineering

features = [
    'value',
    'mean',
    'var',
    # 'mode',
    'skew',
    'std',
    'max',
    'min',
    'median',
    'cv',
    'rolling_mean',
    'rolling_var',
    'rolling_skew',
    'rolling_std',
    'rolling_max',
    'rolling_min',
    'rolling_median',
    'rolling_cv',
    'abs_energy',
    'absolute_sum_of_changes',
    'ar_coefficient',
    'autocorrelation',
    'augmented_dickey_fuller',
    'approximate_entropy',
    'kurtosis',
    'diff_1',
    'diff_24'
]

target = 'label'


def detect(data):
    # kpi_df = get_data_reference(dataset=dataset, dataset_entity=dataset_entity).to_pandas_dataframe()
    # kpis = kpi_df['kpi_id'].value_counts()
    res_list = []
    kpis_list = ["9415ac3c-cae9-4906-b65c-bc9c7a732c30","600a5d6e-fd61-43a9-9857-783cec807879",
                 "31997140-314b-459a-a69c-c9d3e31ec1a1","a113b2a7-0a80-4ef6-8dac-b35ab1ca4f98",
                 "e6999100-d229-41c1-9370-f7b5ff315b8b","3fe4d11f-4e06-4725-bf16-19db40a7a3e1",
                 "681cbb98-68e2-4d9a-af4f-9efde2768a5e","29374201-b68d-4714-a2ee-4772ac52447f",
                 "355eda04-426e-4c9f-aba0-6481db290068","4f4936b1-1a23-4eba-9e69-41a304e9b1a1",
                 "b38421e2-5c20-4734-bdf0-8ab3b8c721a6","21aa1802-ad3e-4dda-b34f-ad3526f6130b",
                 "bb6bb8fb-11a0-45c0-8efd-6c0791700ea0","0528d024-7cb5-4e15-910f-39fb74b68625",
                 "0a9f5909-7690-4ab6-b153-a4be885c29e0","fec34c7e-2298-498e-896e-f0ce7716740d",
                 "eeb90da1-04ce-4bb4-b054-f193cdc72b64","3e1f1faa-a37e-41f2-a49c-dfae19b8f8a0",
                 "8f522bbf-6e5f-4fed-ac59-25b242531305","ed63c9ea-322d-40f7-bbf0-c75fb275c067"]
    for kpi_id in kpis_list:
        # kpi_id = kpis.index[i]
        # kpi_series = df['value'].astype(float)
        # kpis_data = kpi_series.interpolate()
        # df['value'] = kpis_data
        df = data.loc[data['kpi_id'] == kpi_id]
        df = feature_engineering(df, 5, norm=True)
        model_path = os.path.join(
            "./model",  kpi_id+'.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
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
    out_path = os.path.join("result.csv")
    with open(out_path, 'w') as f:
        res_all.to_csv(f, index=False)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    test_df = pd.read_csv("./data/KPI_Test.csv")
    detect(test_df)
    # csv=pd.read_csv("./result.csv")
    # print(csv.head(10))