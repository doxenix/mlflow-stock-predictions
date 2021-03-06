import requests
import json
import numpy as np
import pandas as pd
import joblib
from pipline import get_data
from mlflow.tracking import MlflowClient
from params import hyperparams

hyperparams = hyperparams

CONFIG = [hyperparams]

def prepare_data_for_predicion(config_idx=0):
    config = CONFIG[config_idx]
    client = MlflowClient(registry_uri='sqlite:///mlruns.db')
    uri = client.get_model_version_download_uri("model1", version=1)
    uri = uri.split('/')[3]

    scaler = joblib.load(f'mlruns/1/{uri}/artifacts/scaler.pkl')

    data = get_data(sample=config.sample)
    total_dataset=data[['Close']]
    data_test = data[['Close']].iloc[config.lags:].copy()
    model_inputs=total_dataset[len(total_dataset)-len(data_test)-config.lags:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    real_data = [model_inputs[len(model_inputs)-config.lags:len(model_inputs+1), 0]]
    real_data = np.array(real_data)
    real_data=real_data[...,None]

    data = real_data.reshape(1, -1)
    return data, scaler

if __name__ == '__main__':
    data, scaler = prepare_data_for_predicion()
    data_json = json.dumps(data.tolist())

    headers = {'Content-Type': 'application/json; format=pandas-records'}
    request_uri = 'http://127.0.0.1:5000/invocations'
    try:
        response = requests.post(request_uri, data=data_json, headers=headers)
        dict = response.json()
        pred_value = dict[0]['0']    
        pred_value = scaler.inverse_transform([[pred_value]])
        print((pred_value))        
        print('done!!!')
    except Exception as ex:
        raise (ex)
