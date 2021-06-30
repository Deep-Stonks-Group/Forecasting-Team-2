import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PythonDataProcessing import DataRetrieval as DR
from PythonDataProcessing import Metrics as MET
from pandas.core.frame import DataFrame
from sklearn.preprocessing import MinMaxScaler


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True).float()

        self.fc = nn.Linear(hidden_size, num_classes)

        self.trained_tickers = []

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size) #reshape output from 1,train_len,hidden to train_len,hidden
        out = self.fc(h_out)
        return out

def sliding_windows( data, labels, seq_length):
    x = []
    y = []
    for i in range(data.shape[0]-seq_length-1):
        _x = data[i: (i+seq_length)]
        _y = labels[i+seq_length]
        x.append(_x)
        y.append(_y)
    return x, y

def retrieve_stock_data(ticker: str, input_dims, label_dims) -> DataFrame:
    # Loading Data
    # data_source = DR.get_crypto_data('ETH-BTC',interval='1h',start_date='2021-04-01-00-00')
    data_source = DR.get_stock_data(ticker,interval='1d',period='2y')
    stock_dataframe = deepcopy(data_source)
    new_features = []
    new_features.extend(set(input_dims) - set(data_source.columns))
    new_features.extend(set(label_dims) - set(data_source.columns))

    for new_feature in new_features:
        try:
            stock_dataframe = DR.add_technical_indicators[new_feature](data_source)
        except KeyError:
            print(f'No function exists to add the dimension {new_feature}')
            raise

    return stock_dataframe


def fit_scalers(data_source: DataFrame, train_size, input_dims, label_dims):
    data_scaler = MinMaxScaler()
    labl_scaler = MinMaxScaler()

    data_scaler.fit(data_source[input_dims].iloc[:train_size])
    labl_scaler.fit(data_source[label_dims].iloc[:train_size])

    return data_scaler, labl_scaler


def normalize_stock_data(data_source, data_scaler, labl_scaler, input_dims, label_dims):
    scaled_data = data_scaler.transform(data_source[input_dims])
    scaled_lbls = labl_scaler.transform(data_source[label_dims])
    return scaled_data, scaled_lbls



def train_model(lstm, train_x, train_y, epochs=2000, learning_rate=.01):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    lstm.train()
    for epoch in range(epochs):
        outputs = lstm(train_x)
        optimizer.zero_grad()
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

class PredictionEngine():

    def __init__(self, input_dims=['High','Low','Close','Volume','EMA'], label_dims=['SMA'], seq_length=10):
        self.input_dims = input_dims
        self.label_dims = label_dims
        self.seq_length = seq_length

    def create_model(self, input_size=5, hidden_size=6, num_layers=1, num_classes=1):
        self.lstm = LSTM(num_classes, input_size, hidden_size, num_layers, self.seq_length)

    def save_model(self):
        self.lstm.trained_tickers.sort()
        name = ''.join(self.lstm.trained_tickers)
        with open('simple_lstm/models/' + name + '.p', 'wb') as outfile:
            pickle.dump(self.lstm, outfile)

    def load_model(self, name):
        if name:
            try:
                with open('simple_lstm/models/' + name + '.p', 'rb') as infile:
                    self.lstm = pickle.load(infile)
            except Exception as e:
                print('could not load model {}'.format(name))
                raise

    def train_ticker(self, ticker: str, training_set_coeff=0.8):
        if not self.lstm:
            self.create_model()
        self.lstm.train()
        if ticker in self.lstm.trained_tickers:
            print(f'Already trained model on {ticker}')
            return
        stock_dataframe = retrieve_stock_data(ticker, self.input_dims, self.label_dims)
        train_size = int(len(stock_dataframe) * training_set_coeff)
        data_scaler, label_scaler = fit_scalers(stock_dataframe, train_size, self.input_dims, self.label_dims)
        scaled_data, scaled_labels = normalize_stock_data(stock_dataframe, data_scaler, label_scaler, self.input_dims, self.label_dims)
        x, y = sliding_windows(scaled_data, scaled_labels, self.seq_length)
        train_x = torch.Tensor(x[0:train_size])
        train_y = torch.Tensor(y[0:train_size])
        train_model(self.lstm, train_x, train_y)
        self.lstm.trained_tickers.append(ticker)

    def eval_ticker(self, ticker, training_set_coeff=0.8):
        self.lstm.eval()

        stock_dataframe = retrieve_stock_data(ticker, self.input_dims, self.label_dims)
        train_size = int(len(stock_dataframe) * training_set_coeff)
        data_scaler, label_scaler = fit_scalers(stock_dataframe, train_size, self.input_dims, self.label_dims)
        scaled_data, scaled_labels = normalize_stock_data(stock_dataframe, data_scaler, label_scaler, self.input_dims, self.label_dims)
        
        x, y = sliding_windows(scaled_data, scaled_labels, self.seq_length)
        dataX = torch.Tensor(np.array(x))
        dataY = torch.Tensor(np.array(y))
        test_x = torch.Tensor(x[train_size:])
        test_y = torch.Tensor(y[train_size:])

        all_predict = self.lstm(dataX)
        data_predict = all_predict.data.numpy()
        dataY_plot = dataY.data.numpy()

        data_predict = label_scaler.inverse_transform(data_predict)
        dataY_plot = label_scaler.inverse_transform(dataY_plot.reshape(dataY_plot.shape[0],1))

        plt.axvline(x=train_size, c='r', linestyle='--')
        plt.plot(dataY_plot)
        plt.plot(data_predict)
        plt.suptitle('Time-Series Prediction')
        plt.show()
        MET.print_metrics(test_x,test_y,self.lstm,label_scaler)

    def predict(self, ticker, input_sequence):
        # would like to not fit scalers every time but I dont want to store them anywhere yet
        stock_dataframe = retrieve_stock_data(ticker, self.input_dims, self.label_dims)
        train_size = len(stock_dataframe)
        data_scaler, label_scaler = fit_scalers(stock_dataframe, train_size, self.input_dims, self.label_dims)
        scaled_input_sequence = data_scaler.transform(input_sequence)
        output = self.lstm(scaled_input_sequence)
        prediction = label_scaler.inverse_transform(output.data.numpy)
        return prediction


""" USAGE EXAMPLE """

predictor = PredictionEngine()
predictor.create_model()

""" training on multiple tickers produces far less accurate eval results than one ticker per model

training_set_coeff = 0.8
top_stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'CMCSA', 'JPM', 'HD', 'DIS', 'XOM']
for stock in top_stocks:
    model_1.train_ticker(stock, training_set_coeff)

"""

predictor.train_ticker('GOOGL')
predictor.train_ticker('FB')

predictor.eval_ticker('FB')
predictor.save_model()


