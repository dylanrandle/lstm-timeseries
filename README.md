# LSTM for Timeseries Prediction

Very simple, Vanilla LSTM model to do timeseries prediction. Optimized with SGD over mean-squared error. 

Place data in ```/data``` and point to it with command line args.

To run training: ``` python3 train.py ```

Dependencies: tensorflow, numpy, pandas, six, matplotlib.

Apply -h flag to see available args.

To run Tensorboard: ``` tensorboard --logdir=./logs/ ``` 

This was made as I was initially learning how to use Tensorflow. TQQQ is a highly volatile leveraged ETF, and thus the predictions are not very good with this simple model. 
