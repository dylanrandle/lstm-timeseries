# LSTM for Timeseries Prediction of TQQQ

Very simple LSTM model to do timeseries prediction. Optimized with SGD over mean-squared error. 

To run training: ``` python3 train.py ```

To run Tensorboard: ``` tensorboard --logdir=./logs/ ``` 

This was made as I was initially learning how to use Tensorflow. 

Future:
  * Get more data
  * Implement gradient clipping and exponential learning rate decay
  * Visualize predictions
  * Tune hyperparameters
