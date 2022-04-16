from sklearn.preprocessing import MinMaxScaler
from argparse import Namespace

hyperparams = Namespace(
    scaler=MinMaxScaler(feature_range=(0,1)),
    batch_size = 64, 
    learning_rate=0.001,    
    units_layer1=50,
    units_layer2=50,
    units_layer3=50,
    units_layer4=1,
    dropout_size=0.2,
    epochs=27,
    data_split = 0.66,
    lags=60,
    ticker='PLN=X',
    start='2012-01-02',
    end='2022-04-14',
    plot_predictions = True
)