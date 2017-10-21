# LSTM for Timeseries Prediction of TQQQ

LSTM model to do timeseries prediction. Optimized with SGD over mean-squared error. 

To run training: ``` python3 train.py ``` 

Apply -h flag for help with arguments. 

Available args:
  * --data_file=
  * --save_dir=
  * --save_every=
  * --log_dir=
  * --seq_len=
  * --hidden_dim=
  * --batch_size=
  * --learning_rate=
  * --num_epochs=

To run Tensorboard: ``` tensorboard --logdir=./logs/ ``` 

This was made as I was initially learning how to use Tensorflow. TQQQ is a highly volatile leveraged ETF, and thus the predictions are not very good with this simple model. 

Future work:
  * Implement gradient clipping and exponential learning rate decay
  * Visualize predictions
  * Tune hyperparameters
  * Get more data
