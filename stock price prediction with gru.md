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






## Third attempt
- weird results investigate further


Epoch 0/75 completee
----------------
MSE: 0.0679953
MAE: 0.259794
RMSE: 0.260759
Profit/Loss: 5.32814%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.0243959%
----------------
Epoch 1/75 completee
----------------
MSE: 0.000843355
MAE: 0.0246838
RMSE: 0.0290406
Profit/Loss: 0.626862%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00286323%
----------------
Epoch 2/75 completee
----------------
MSE: 0.000559714
MAE: 0.0190239
RMSE: 0.0236583
Profit/Loss: 0.425388%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.0019427%
----------------
Epoch 3/75 completee
----------------
MSE: 0.000526912
MAE: 0.0182723
RMSE: 0.0229546
Profit/Loss: 0.394263%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00180053%
----------------
Epoch 4/75 completee
----------------
MSE: 0.000514711
MAE: 0.0179809
RMSE: 0.0226872
Profit/Loss: 0.381925%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00174418%
----------------
Epoch 5/75 completee
----------------
MSE: 0.000506037
MAE: 0.017773
RMSE: 0.0224953
Profit/Loss: 0.372865%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00170279%
----------------
Epoch 6/75 completee
----------------
MSE: 0.000498525
MAE: 0.0175895
RMSE: 0.0223277
Profit/Loss: 0.364811%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00166601%
----------------
Epoch 7/75 completee
----------------
MSE: 0.000491672
MAE: 0.0174182
RMSE: 0.0221737
Profit/Loss: 0.357289%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00163165%
----------------
Epoch 8/75 completee
----------------
MSE: 0.000485566
MAE: 0.0172674
RMSE: 0.0220356
Profit/Loss: 0.350435%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00160035%
----------------
Epoch 9/75 completee
----------------
MSE: 0.000480054
MAE: 0.017133
RMSE: 0.0219101
Profit/Loss: 0.344112%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00157147%
----------------
Epoch 10/75 complete
----------------
MSE: 0.000475104
MAE: 0.017011
RMSE: 0.0217969
Profit/Loss: 0.338321%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00154502%
----------------
Epoch 11/75 complete
----------------
MSE: 0.000470466
MAE: 0.0168978
RMSE: 0.0216902
Profit/Loss: 0.33279%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00151975%
----------------
Epoch 12/75 complete
----------------
MSE: 0.000466319
MAE: 0.0167963
RMSE: 0.0215944
Profit/Loss: 0.327757%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00149677%
----------------
Epoch 13/75 complete
----------------
MSE: 0.000462552
MAE: 0.0167046
RMSE: 0.021507
Profit/Loss: 0.323107%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00147553%
----------------
Epoch 14/75 complete
----------------
MSE: 0.000459223
MAE: 0.0166224
RMSE: 0.0214295
Profit/Loss: 0.318934%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00145647%
----------------
Epoch 15/75 complete
----------------
MSE: 0.000456001
MAE: 0.0165419
RMSE: 0.0213542
Profit/Loss: 0.314838%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00143777%
----------------
Epoch 16/75 complete
----------------
MSE: 0.000452914
MAE: 0.0164647
RMSE: 0.0212818
Profit/Loss: 0.310853%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00141956%
----------------
Epoch 17/75 complete
----------------
MSE: 0.000450115
MAE: 0.0163945
RMSE: 0.0212159
Profit/Loss: 0.307191%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00140284%
----------------
Epoch 18/75 complete
----------------
MSE: 0.000447497
MAE: 0.0163284
RMSE: 0.0211541
Profit/Loss: 0.30372%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00138699%
----------------
Epoch 19/75 complete
----------------
MSE: 0.000445121
MAE: 0.0162678
RMSE: 0.0210979
Profit/Loss: 0.30053%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00137242%
----------------
Epoch 20/75 complete
----------------
MSE: 0.000442793
MAE: 0.0162084
RMSE: 0.0210426
Profit/Loss: 0.297367%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00135797%
----------------
Epoch 21/75 complete
----------------
MSE: 0.000440782
MAE: 0.0161568
RMSE: 0.0209948
Profit/Loss: 0.294601%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00134534%
----------------
Epoch 22/75 complete
----------------
MSE: 0.000438727
MAE: 0.0161034
RMSE: 0.0209458
Profit/Loss: 0.291744%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00133229%
----------------
Epoch 23/75 complete
----------------
MSE: 0.000436885
MAE: 0.0160551
RMSE: 0.0209018
Profit/Loss: 0.289157%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00132047%
----------------
Epoch 24/75 complete
----------------
MSE: 0.000435102
MAE: 0.0160079
RMSE: 0.0208591
Profit/Loss: 0.286627%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00130892%
----------------
Epoch 25/75 complete
----------------
MSE: 0.000433444
MAE: 0.0159635
RMSE: 0.0208193
Profit/Loss: 0.284251%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00129807%
----------------
Epoch 26/75 complete
----------------
MSE: 0.000431805
MAE: 0.0159192
RMSE: 0.0207799
Profit/Loss: 0.281883%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00128726%
----------------
Epoch 27/75 complete
----------------
MSE: 0.000430285
MAE: 0.0158778
RMSE: 0.0207433
Profit/Loss: 0.279668%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00127714%
----------------
Epoch 28/75 complete
----------------
MSE: 0.00042882
MAE: 0.0158375
RMSE: 0.020708
Profit/Loss: 0.277517%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00126731%
----------------
Epoch 29/75 complete
----------------
MSE: 0.00042742
MAE: 0.0157986
RMSE: 0.0206741
Profit/Loss: 0.275443%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00125784%
----------------
Epoch 30/75 complete
----------------
MSE: 0.000426079
MAE: 0.0157615
RMSE: 0.0206417
Profit/Loss: 0.27344%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.0012487%
----------------
Epoch 31/75 complete
----------------
MSE: 0.000424747
MAE: 0.0157247
RMSE: 0.0206094
Profit/Loss: 0.271437%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00123955%
----------------
Epoch 32/75 complete
----------------
MSE: 0.000423511
MAE: 0.0156907
RMSE: 0.0205794
Profit/Loss: 0.269561%
Direction Accuracy: 54.3379%
Total Trades: 219
Avg Trade Return: 0.00123098%
----------------
Epoch 33/75 complete
----------------
MSE: 0.0004223
MAE: 0.0156573
RMSE: 0.0205499
Profit/Loss: 0.248383%
Direction Accuracy: 56.4356%
Total Trades: 101
Avg Trade Return: 0.00245863%
----------------
Epoch 34/75 complete
----------------
MSE: 0.000421124
MAE: 0.0156245
RMSE: 0.0205213
Profit/Loss: 0.0124046%
Direction Accuracy: 61.1111%
Total Trades: 18
Avg Trade Return: 0.000692033%
----------------
Epoch 35/75 complete
----------------
MSE: 0.000419998
MAE: 0.0155929
RMSE: 0.0204939
Profit/Loss: 0.0347941%
Direction Accuracy: 83.3333%
Total Trades: 6
Avg Trade Return: 0.00579972%
----------------
Epoch 36/75 complete
----------------
MSE: 0.000418886
MAE: 0.0155619
RMSE: 0.0204667
Profit/Loss: 0%
Direction Accuracy: 0%
Total Trades: 0
Avg Trade Return: 0%
----------------
Epoch 37/75 complete
----------------
MSE: 0.000417763
MAE: 0.015531
RMSE: 0.0204393
Profit/Loss: 0%
Direction Accuracy: 0%
Total Trades: 0
Avg Trade Return: 0%
----------------
Epoch 38/75 complete
----------------
MSE: 0.000416737
MAE: 0.0155026
RMSE: 0.0204141
Profit/Loss: 0%
Direction Accuracy: 0%
Total Trades: 0
Avg Trade Return: 0%
----------------
Epoch 39/75 complete
----------------
MSE: 0.000415735
MAE: 0.0154747
RMSE: 0.0203896
Profit/Loss: 0%
Direction Accuracy: 0%
Total Trades: 0
Avg Trade Return: 0%
----------------
Epoch 40/75 complete
----------------
MSE: 0.000414729
MAE: 0.0154466
RMSE: 0.0203649
Profit/Loss: 0%
Direction Accuracy: 0%
Total Trades: 0
Avg Trade Return: 0%
----------------
Epoch 41/75 complete
----------------
MSE: 0.000413812
MAE: 0.0154212
RMSE: 0.0203424
Profit/Loss: 0%
Direction Accuracy: 0%
Total Trades: 0
Avg Trade Return: 0%
----------------
Epoch 42/75 complete
----------------
MSE: 0.000412844
MAE: 0.0153942
RMSE: 0.0203186
Profit/Loss: 0%
Direction Accuracy: 0%
Total Trades: 0
Avg Trade Return: 0%
----------------
Epoch 43/75 complete
----------------
MSE: 0.000411987
MAE: 0.0153702
RMSE: 0.0202975
Profit/Loss: 0%
Direction Accuracy: 0%
Total Trades: 0
Avg Trade Return: 0%
----------------
Epoch 44/75 complete
----------------
MSE: 0.000411095
MAE: 0.015345
RMSE: 0.0202755
Profit/Loss: 0%
Direction Accuracy: 0%
Total Trades: 0
Avg Trade Return: 0%
----------------
Epoch 45/75 complete
----------------
MSE: 0.000410202
MAE: 0.0153201
RMSE: 0.0202534
Profit/Loss: 0%
Direction Accuracy: 0%
Total Trades: 0
Avg Trade Return: 0%
----------------
Epoch 46/75 complete
----------------
MSE: 0.000409342
MAE: 0.015296
RMSE: 0.0202322
Profit/Loss: 0%
Direction Accuracy: 0%
Total Trades: 0
Avg Trade Return: 0%
----------------
Epoch 47/75 complete
----------------
MSE: 0.000408522
MAE: 0.0152728
RMSE: 0.0202119
Profit/Loss: 0%
Direction Accuracy: 0%
Total Trades: 0
Avg Trade Return: 0%
----------------
Epoch 48/75 complete
----------------
MSE: 0.000407663
MAE: 0.0152484
RMSE: 0.0201907
Profit/Loss: 0%
Direction Accuracy: 0%
Total Trades: 0
Avg Trade Return: 0%
----------------
Epoch 49/75 complete
----------------
MSE: 0.000406883
MAE: 0.0152261
RMSE: 0.0201713
Profit/Loss: 0%
Direction Accuracy: 0%
Total Trades: 0
Avg Trade Return: 0%
----------------
Epoch 50/75 complete
----------------
MSE: 0.000406056
MAE: 0.0152022
RMSE: 0.0201508
Profit/Loss: 0%
Direction Accuracy: 0%
Total Trades: 0
Avg Trade Return: 0%
----------------
Epoch 51/75 complete
----------------
MSE: 0.000405277
MAE: 0.0151801
RMSE: 0.0201315
Profit/Loss: 0%
Direction Accuracy: 0%
Total Trades: 0
Avg Trade Return: 0%
----------------
Epoch 52/75 complete
----------------
MSE: 0.00040456
MAE: 0.01516
RMSE: 0.0201137
Profit/Loss: 0%
Direction Accuracy: 0%
Total Trades: 0
Avg Trade Return: 0%
----------------
Epoch 53/75 complete
----------------
MSE: 0.000403803
MAE: 0.0151387
RMSE: 0.0200949
Profit/Loss: 0%
Direction Accuracy: 0%
Total Trades: 0
Avg Trade Return: 0%
----------------
Epoch 54/75 complete
----------------
MSE: 0.000403033
MAE: 0.0151168
RMSE: 0.0200757
Profit/Loss: 0%
Direction Accuracy: 0%
Total Trades: 0
Avg Trade Return: 0%
----------------
Epoch 55/75 complete
----------------
MSE: 0.000402325
MAE: 0.0150966
RMSE: 0.020058
Profit/Loss: 0%
Direction Accuracy: 0%
Total Trades: 0
Avg Trade Return: 0%
----------------
Epoch 56/75 complete
----------------
MSE: 0.000401638
MAE: 0.0150769
RMSE: 0.0200409
Profit/Loss: 0%
Direction Accuracy: 0%
Total Trades: 0
Avg Trade Return: 0%
----------------
Epoch 57/75 complete
----------------
MSE: 0.000400952
MAE: 0.0150574
RMSE: 0.0200238
Profit/Loss: 0%
Direction Accuracy: 0%
Total Trades: 0
Avg Trade Return: 0%
----------------
Batch 0/17 complete  