import mlflow
import mlflow.tensorflow
from mlflow.models.signature import infer_signature

import matplotlib.pyplot as plt

import os
import datetime

import pandas as pd
import numpy as np

import yfinance as yf

from params import hyperparams
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam

EXP_NAME = 'USD_PLN_daily'
EXP_ID = 1

hyperparams = hyperparams

CONFIG = [hyperparams]

def get_data(config_idx=0, sample=None):
    config = CONFIG[config_idx]

    ticker = yf.Ticker(config.ticker)
    
    if sample is None:           
        data = ticker.history(start=config.start, end=config.end)        
    else:
        data = ticker.history(period=sample)
    return data

def modelDNN(input_shape, config_idx=0):

    config = CONFIG[config_idx]

    model = Sequential()

    model.add(LSTM(units=config.units_layer1, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(config.dropout_size))
    model.add(LSTM(units=config.units_layer2, return_sequences=True))
    model.add(Dropout(config.dropout_size))
    model.add(LSTM(units=config.units_layer3))
    model.add(Dropout(config.dropout_size))
    model.add(Dense(units=config.units_layer4))

    adam = Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['MeanSquaredError'])   

    return (model, "Model 1")

def prepare_train_data(data, config_idx=0):

    usd = data
    config = CONFIG[config_idx]
    
    split = int(len(usd) * config.data_split)
    usd_train = usd[['Close']].iloc[:split].copy()

    scaler=config.scaler
    scaled_data=scaler.fit_transform(usd_train.values.reshape(-1,1))
    prediction_lags = config.lags

    x_train=[]
    y_train=[]

    for x in range(prediction_lags, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_lags:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return (x_train, y_train)

def prepare_test_data(data, config_idx=0):

    usd = data
    config = CONFIG[config_idx]

    split = int(len(usd) * config.data_split)

    usd_test = usd[['Close']].iloc[split:].copy()
    actual_prices=usd_test.values

    total_dataset=usd[['Close']]

    model_inputs=total_dataset[len(total_dataset)-len(usd_test)-config.lags:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = config.scaler.transform(model_inputs)

    # Make Predictions on Test Data
    x_test=[]

    for x in range(config.lags, len(model_inputs)):
        x_test.append(model_inputs[x-config.lags:x, 0])

    x_test=np.array(x_test)
    x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))   

    return x_test, actual_prices

def predict(model, x_test, get_signature=False, config_idx=0):

    config = CONFIG[config_idx]

    result=model.predict(x_test)
    result=config.scaler.inverse_transform(result)

    if get_signature:
        return infer_signature(model_input=x_test, model_output=result), result

def train(model, x_train, y_train, x_test, actual_prices, config_idx=0):    

    config = CONFIG[config_idx]    

    #saving the model name based on timestamp value
    model_path = "models/{:%d-%b-%y_%H-%M-%S}".format(datetime.datetime.now())
    
    
    print("Saving model at ", model_path)
    st = datetime.datetime.now()
    history = model.fit(x_train, y_train, epochs=config.epochs, batch_size=config.batch_size)

    end = datetime.datetime.now()
    time_taken = (end - st).seconds

    signature, result = predict(model, x_test, get_signature=True)    
    
    metric = 'loss'
    mlflow.log_metric(metric, history.history[metric][-1])    
    
    #Additional information to log
    mlflow.log_param("learning_rate", config.learning_rate)
    mlflow.log_param("epochs", config.epochs)
    mlflow.log_param("time_taken", time_taken)
    
    mlflow.log_param("model_path", model_path)    
    
    #Save the model under keras    
    mlflow.keras.save_model(model,
                            model_path,
                            # python_model=loader_mod.MyPredictModel(path),
                            signature=signature)
    pickle.dump(config.scaler, open(f'{model_path}/scaler.pkl', 'wb'))
    mlflow.log_artifact(f'{model_path}/scaler.pkl')
                            
    if config.plot_predictions:
    #save predictions to chart jpg                       
        plt.plot(actual_prices, color = "black", label="Actual Price")
        plt.plot(result, color="green", label="Predicted Price")
        plt.title("USD_PLN Price")
        plt.xticks([])
        plt.legend()
        plt.savefig(f'{model_path}/usd_pln_prediction_fig.jpg')
        mlflow.log_artifact(f'{model_path}/usd_pln_prediction_fig.jpg')

def run(config_idx=0):
    mlflow.set_tracking_uri("sqlite:///mlruns.db")

    data = get_data()
    x_train, y_train = prepare_train_data(data)
    x_test, actual_prices = prepare_test_data(data)

    input_shape=(x_train.shape[1], 1)
    mlflow.tensorflow.autolog()    
    
    model, model_name = modelDNN(input_shape=input_shape, config_idx=0)

    print(model_name)
    print(model.summary())

    run_name = f"Experiment: Pet Classifier MODEL CONFIG {config_idx}"
    #For each run setup in the following manner to let mlflow know when you
    #train the model.
    
    with mlflow.start_run(run_name=run_name, experiment_id=EXP_ID) as run:

        run_id = run.info.run_uuid
        
        print(f"*****Running Run {run_id} *****")
        mlflow.log_param("Experiment Name", run_name)
        train(model, x_train, y_train, x_test, actual_prices, config_idx=0)

if __name__ == "__main__": 
    run()
        





