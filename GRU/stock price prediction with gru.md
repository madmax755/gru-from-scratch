## First attempt
- GRU Cell with linear output layer.
    - GRU/stock_data/AAPL_data_normalised.csv training/test data (14 seq length then 80/20 split)
    - 5 input features (return, volume return, day volatility, RSI normalised, MACD normalised)
    - 14 day sequence length
    - 64 hidden size
    - 1 output feature (next day return)
    - 50 epochs
    - convergence to 0.0005 MSE after ~30 epochs

    - seemingly random predictions
        - with proportional bet sizing with a 1% return threshold
        - ~200 trades
        - ~51% directional accuracy
        - ~0.1% return per trade
        - small test set so don't put too much stock in these results


## Second attempt
- GRU Cell with MLP output layer.
    - GRU/stock_data/AAPL_data_normalised.csv training/test data (100 seq length then 80/20 split)
    - 5 input features (return, volume return, day volatility, RSI normalised, MACD normalised)
    - 128 hidden size
    - 128, 32, 1 MLP topology
    - sigmoid activation functions for all layers including output layer!!!!! (probably not a good idea!!!)
    - 1 output feature (next day return)
    - 75 epochs (still converging but slowly)
    - convergence to ~0.0005 MSE

    - predictions were all ~0.016 (i.e. 1.6% return)
        - MSE: 0.000589931
        - MAE: 0.0193555
        - RMSE: 0.0242885
        - Profit/Loss: -0.262597%
        - Direction Accuracy: 51.5982%
        - Total Trades: 219
        - Avg Trade Return: -0.00119666%
        - Sample predictions:
            - Predicted: 0.016214 Actual: 0.007447
            - Predicted: 0.0162048 Actual: 0.006287
            - Predicted: 0.0161902 Actual: -0.001983
            - Predicted: 0.016214 Actual: 0.010217
            - Predicted: 0.0162136 Actual: 0.010205
            - Predicted: 0.0162262 Actual: 0.018632
            - Predicted: 0.016199 Actual: 0.012028
            - Predicted: 0.016212 Actual: -0.018169
            - Predicted: 0.0162202 Actual: -0.030252
            - Predicted: 0.0161988 Actual: -0.004122

        - rough estimate for the mean of the train set is 0.0008
        - odd as would expect a result like the network just predicting the mean of the train set