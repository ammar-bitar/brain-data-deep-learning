from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Conv2D, MaxPool2D, LSTM
from tensorflow.keras.layers import Lambda,Activation
from tensorflow.keras.layers import dot
import tensorflow as tf


"""# Cascade Attention model"""

class AttentionCascadeMulti:
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

    
    def attention_block(self,hidden_states):
        hidden_size = int(hidden_states.shape[2])
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
        score = dot([score_first_part, h_t], [2, 1], name='attention_score')
        attention_weights = Activation('softmax', name='attention_weight')(score)
        context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
        pre_activation = concatenate([context_vector, h_t], name='attention_output')
        attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        return attention_vector
        
        
    def cascade_model(self):
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
              conv = Conv2D(self.number_conv2D_filters, self.kernel_shape, activation=self.cnn_activation, input_shape=(self.mesh_rows, self.mesh_columns,1))(inputs[i])# modify shape and kernel
              pool = MaxPool2D(pool_size=self.pool_size)(conv) # modify pool size
              convs.append(Flatten()(pool))
    
      merge = concatenate(convs)
      merge = tf.expand_dims(merge,axis=1)
      lstm = LSTM(self.number_lstm_cells, return_sequences=True)(merge)
      attention_output = self.attention_block(lstm)
      hidden1 = Dense(self.number_nodes_hidden, activation=self.hidden_activation)(attention_output)
      output = Dense(self.number_classes, activation=self.model_activation)(hidden1)
      model = Model(inputs=inputs, outputs=output)
      return model


