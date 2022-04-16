import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime,date
import matplotlib.pyplot as plt
from itertools import product
plt.style.use("seaborn")

PATH = r'C:\Users\domin\Desktop\Nauka\stock\oanda.cfg'
api = tpqoa.tpqoa(PATH)

class SMABacktester():

    def __init__(self, symbol, sma_s, sma_l, start, end):
        self._symbol = symbol
        self.sma_s = sma_s
        self.sma_l = sma_l
        self.start = start
        self.end = end
        self.results = None
        self.get_data()
        self.prepare_data()

    def get_intruments(self, sort=True):
        get_instruments = api.get_instruments()
        df = pd.DataFrame.from_records(self.data, columns=['Name', 'Pair'])
        if sort == True:
            df = df.sort_values(by='Pair').reset_index()
        return df

    def get_data(self, granularity="D", price="B"):
        raw = api.get_history(instrument = self._symbol, start=self.start, end=self.end,
                granularity=granularity, price=price)
        raw.index = raw.index.date
        raw.index.name = "Date"
        raw['price'] = raw['c']
        raw.drop(columns=['volume', 'o', 'h', 'l', 'c', 'complete'], axis=1, inplace=True)
        raw['returns'] = np.log(raw.price / (raw.price.shift(1)))        
        self.data = raw
        

    def prepare_data(self):
        '''Prepares the data for strategy backtesting (strategy-specific).
        '''
        data = self.data.copy()
        data["SMA_S"] = data["price"].rolling(self.sma_s).mean()
        data["SMA_L"] = data["price"].rolling(self.sma_l).mean()
        self.data = data

    def set_parameters(self, sma_s=None, sma_l=None):
        if sma_s is not None:
            self.sma_s = sma_s
            self.data["SMA_S"] = self.data.price.rolling(self.sma_s).mean()
        if sma_l is not None:
            self.sma_l = sma_l
            self.data["SMA_L"] = self.data.price.rolling(self.sma_s).mean()

    def test_strategy(self):
        data = self.data.copy().dropna()
        data['position'] = np.where(data["SMA_S"] > data["SMA_L"], 1,-1)
        data['strategy'] = data['position'].shift(1) * data['returns']
        data.dropna(inplace=True)
        data['creturns'] = data['returns'].cumsum().apply(np.exp)
        data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
        self.results = data
        
        perf = data['cstrategy'].iloc[-1]
        outperf = perf - data['creturns'].iloc[-1]
        return round(perf, 6), round(outperf, 6)

    def plot_results(self):
        if self.results is None:
            print('test_strategy() has not been executed. No data to generate the plot')
        else:
            self.results[['creturns', 'cstrategy']].plot(figsize=(12,8), 
                                                        title=f'{self._symbol} | SMA_S = {self.sma_s} | SMA_L = {self.sma_l}')

    def optimize_parameters(self, SMA_S_range, SMA_L_range):
        ''' Finds the optimal strategy (global maximum) given the SMA parameter ranges.

        Parameters
        ----------
        SMA_S_range, SMA_L_range: tuple
            tuples of the form (start, end, step size)
        '''
        
        combinations = list(product(range(*SMA_S_range), range(*SMA_L_range)))
        
        # test all combinations
        results = []
        for comb in combinations:
            self.set_parameters(comb[0], comb[1])
            results.append(self.test_strategy()[0])
        
        best_perf = np.max(results) # best performance
        opt = combinations[np.argmax(results)] # optimal parameters
        
        # run/set the optimal strategy
        self.set_parameters(opt[0], opt[1])
        self.test_strategy()
                   
        # create a df with many results
        many_results =  pd.DataFrame(data = combinations, columns = ["SMA_S", "SMA_L"])
        many_results["performance"] = results
        self.results_overview = many_results
                            
        return opt, best_perf


            