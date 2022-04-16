import requests
import json
import numpy as np
import pandas as pd
import joblib
from pipline import get_data
from mlflow.tracking import MlflowClient

client = MlflowClient(registry_uri='sqlite:///mlruns.db')
uri = client.get_model_version_download_uri("model1", version=1)
uri = uri.split('/')[3]

scaler = joblib.load(f'mlruns/1/{uri}/artifacts/scaler.pkl')

usd = get_data(sample='100d')
total_dataset=usd[['Close']]
usd_test = usd[['Close']].iloc[60:].copy()
model_inputs=total_dataset[len(total_dataset)-len(usd_test)-60:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

real_data = [model_inputs[len(model_inputs)-60:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data=real_data[...,None]

data = real_data.reshape(1, -1)

data_json = json.dumps(data.tolist())
# print(data_json)
headers = {'Content-Type': 'application/json; format=pandas-records'}
request_uri = 'http://127.0.0.1:5000/invocations'


if __name__ == '__main__':
    try:
        response = requests.post(request_uri, data=data_json, headers=headers)
        dict = response.json()
        pred_value = dict[0]['0']    
        pred_value = scaler.inverse_transform([[pred_value]])
        print((pred_value))
        print('done!!!')
    except Exception as ex:
        raise (ex)