from sklearn.preprocessing import MinMaxScaler
from argparse import Namespace

hyperparams = Namespace(
    scaler=MinMaxScaler(feature_range=(0,1)),
    batch_size=64, 
    learning_rate=0.001,    
    units_layer1=50,
    units_layer2=50,
    units_layer3=50,
    units_layer4=1,
    dropout_size=0.22,
    epochs=27,
    data_split = 0.7, #size of spliting train/test data
    lags=61, #use in preparing data for model. Number of previous days/hours in one sample of data
    ticker='PLN=X',
    start='2012-01-02', #use in preparing data for model
    end='2022-04-14', #use in preparing data for model
    sample='100d', #use in test data in run_prediction
    plot_predictions=True,
    EXP_NAME='USD_PLN_daily',
    EXP_ID=1    
)