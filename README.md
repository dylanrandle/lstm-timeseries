# LSTM for Timeseries Prediction of TQQQ

Very simple LSTM model to do timeseries prediction. Optimized with SGD over mean-squared error. 

To run training: ``` python3 train.py ```

To run Tensorboard: ``` tensorboard --logdir=./logs/ ``` 

This was made as I was initially learning how to use Tensorflow. 

TODO:
  * Get more data
  * Implement gradient clipping, learning rate exponential decay
  * Visualize predictions
