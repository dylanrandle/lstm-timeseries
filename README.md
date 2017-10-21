# LSTM for Timeseries Prediction

LSTM model to do timeseries prediction. Optimized with SGD over mean-squared error. 

Data in ```/data```.

To run training: ``` python3 train.py ```

There are quite a few dependencies, including Tensorflow of course. 

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