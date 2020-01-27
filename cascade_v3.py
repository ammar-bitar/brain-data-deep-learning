import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, LSTM, Dropout, concatenate, Flatten, Dense, Input
import tensorflow

class Cascade:
    def __init__(self):
        self.model = self.cascade_model()

    def cascade_model(self):
      inputs = []
      convs = []
      for i in range(5):
          input_layer = Input(shape=(20, 22, 1), name = "input"+str(i+1))
          inputs.append(input_layer)

      for i in range(5):
          conv1 = Conv2D(16, (6,6), padding = 'same',activation="relu",input_shape=(20,22,1))(inputs[i])
          
          conv2 = Conv2D(32, (6,6), padding = 'same', activation="relu")(conv1)
          
          conv3 = Conv2D(64, (6,6), padding = 'same', activation="relu")(conv2)

          conv3 = Dropout(0.3)(conv3)
          dense = Dense(512, activation="relu")(Flatten()(conv3))
          
          convs.append(dense)
      
      merge = concatenate(convs)
      merge = tensorflow.expand_dims(merge,axis=1)
      lstm1 = LSTM(64, return_sequences=True)(merge)
      lstm2 = LSTM(64, return_sequences=False)(lstm1)
      final = Dropout(0.3)(lstm2)
      output = Dense(4, activation="softmax")(final)
      
      model = Model(inputs=inputs, outputs=output)
      return model