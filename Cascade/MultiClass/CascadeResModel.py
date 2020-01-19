# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 02:43:17 2020

@author: Smail
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Conv2D, MaxPool2D, LSTM
from tensorflow.keras.layers import add
import tensorflow as tf


class CascadeResMulti:
    def __init__(self, window_size, cnn_activation, hidden_activation, model_activation, pool_size,
                 number_conv2D_filters, kernel_shape, number_lstm_cells, number_nodes_hidden, loss,
                 optimizer, using_gpu):
        self.window_size = window_size
        self.cnn_activation = cnn_activation
        self.hidden_activation = hidden_activation
        self.model_activation = model_activation
        self.pool_size = pool_size
        self.number_conv2D_filters = number_conv2D_filters
        self.kernel_shape = kernel_shape
        self.number_lstm_cells = number_lstm_cells
        self.number_nodes_hidden = number_nodes_hidden
        self.mesh_rows = 20
        self.mesh_columns = 22
        self.loss = loss
        self.optimizer = optimizer
        self.using_gpu = using_gpu
        self.number_classes = 4
        self.model = self.cascade_model()

    def residual_connection(self,input_net, i):
        conv = Conv2D(self.number_conv2D_filters, self.kernel_shape, activation=self.cnn_activation, input_shape=(self.mesh_rows, self.mesh_columns,1))(input_net)# modify shape and kernel
        pool = MaxPool2D(pool_size=self.pool_size)(conv)
        flat = Flatten()(pool)
        connection = add([input_net,flat],name="res"+str(i+1))
        return connection

        
        
    def cascade_res_model(self):
      inputs = []
      convs = []
      for i in range(self.window_size):
          input_layer = Input(shape=(self.mesh_rows, self.mesh_columns,1), name = "input"+str(i+1))
          inputs.append(input_layer)
      
      if self.using_gpu:
          for i in range(self.window_size):
              conv = Conv2D(self.number_conv2D_filters, self.kernel_shape, activation=self.cnn_activation, input_shape=(self.mesh_rows, self.mesh_columns,1))(inputs[i])# modify shape and kernel
              convs.append(Flatten()(conv))
      else:
          for i in range(self.window_size):
                convs.append(self.residual_connection(inputs[i],i))

      merge = concatenate(convs)
    #   flatten = Flatten()(merge)
    #   merge = tf.expand_dims(merge,axis=1)
      hidden0 = Dense(100,activation="sigmoid")(merge)
      flatten = Flatten()(hidden0)
      flatten = tf.expand_dims(flatten,axis=1)
      lstm = LSTM(self.number_lstm_cells, return_sequences=False)(flatten)
      hidden1 = Dense(self.number_nodes_hidden, activation=self.hidden_activation)(lstm)
      output = Dense(self.number_classes, activation=self.model_activation)(hidden1)
      model = Model(inputs=inputs, outputs=output)
      return model