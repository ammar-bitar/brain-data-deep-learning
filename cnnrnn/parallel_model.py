# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 00:33:54 2019

@author: Smail
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input, LSTM, Convolution2D, Embedding, Reshape, Concatenate
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model
import numpy as np

window_size = 5
mesh_rows = 20
mesh_columns = 22
cnn_activation ="relu"
lstm_activation="relu"
model_activation="softmax"
pool_size = (1,1)
number_conv2D_filters = 10
kernel_shape = (7)
embedding_output_dim = 256
number_lstm_cells = 10
number_nodes_hidden = 10
num_channels = 128

def cnn_model():
  inputs = []
  models = []
  for i in range(window_size):  # Adding mesh inputs
    inp = Input(shape=(mesh_rows, mesh_columns, 1), name = "input-mesh"+str(i+1))
    inputs.append(inp)
    
  for i in range(window_size): # Connecting mesh inputs to CNNs
    conv = Conv2D(number_conv2D_filters, kernel_shape, activation=cnn_activation, \
                  input_shape=(mesh_rows, mesh_columns, 1))(inputs[i])# modify shape and kernel
    pool = MaxPool2D(pool_size=pool_size)(conv) # modify pool size
    x = Model(inputs=inputs[i], outputs=Flatten()(pool))
    models.append(x)

  merged = concatenate([model.output for model in models])

  output = Dense(1, activation=model_activation)(merged)

  cnn_model = Model(inputs=[model.input for model in models], outputs=output)
  # print(cnn_model.summary())
  # plot_model(cnn_model, show_shapes=True, dpi=50)
  return cnn_model,[model.input for model in models]

def lstm_model():
  models = []
  for i in range(5):
    input = Input(shape=(num_channels, 1))
    x = Dense(8, activation="relu")(input)
    lstm = LSTM(10,activation="relu")(x)
    x = Model(inputs=input, outputs=lstm)
    models.append(x)

  combined = concatenate([model.output for model in models])
  z = Dense(50, activation="relu")(combined)

  lstm_model = Model(inputs=[model.input for model in models], outputs=z)
  return lstm_model,[model.input for model in models]

lstm_model,lstm_inputs = lstm_model()
cnn,cnn_inputs = cnn_model()

combined = concatenate([lstm_model.output,cnn.output])
final = Dense(1,activation="relu")(combined)
inputs_parallel = []
for i in range(len(lstm_inputs)):
  inputs_parallel.append(lstm_inputs[i])
for i in range(len(cnn_inputs)):
  inputs_parallel.append(cnn_inputs[i])

parallel_model = Model(inputs = inputs_parallel,outputs=final)
print(parallel_model.summary())
plot_model(parallel_model, show_shapes=True, dpi=100,to_file='Parallel_CNN_LSTM.png')