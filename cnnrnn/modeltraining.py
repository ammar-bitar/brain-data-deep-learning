# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 00:45:15 2019

@author: Smail
"""
##With Tensorflow 2 ###
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import MaxPool2D, LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Reshape
import numpy as np
#from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers


def custom_test_model():
  # first input model
  visible1 = Input(shape=(64,64,1))
  conv11 = Conv2D(32, kernel_size=4, activation='relu')(visible1)
  pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
  flat1 = Flatten()(pool11)
  # second input model
  visible2 = Input(shape=(32,32,3))
  conv21 = Conv2D(32, kernel_size=4, activation='relu')(visible2)
  pool21 = MaxPooling2D(pool_size=(2, 2))(conv21)
  flat2 = Flatten()(pool21)
  # merge input models

  #print(tf.shape(flat1))
  #print(flat1.shape)
  merge = concatenate([flat1, flat2])
  #tf.reshape(merge, [tf.shape(merge), -1])
  #print(merge.get_shape()[0])
  #print("rank of merge",tf.rank(merge))
  #print(flat1.get_shape())
  print(flat1.shape.as_list()[1:])
  #merge = tf.reshape(merge,(tf.shape(merge)[0],tf.shape(merge)[1],1))
  embed = Embedding(input_dim=35072, output_dim=256, input_length=35072)(merge)
  #embed = Embedding(input_dim=, output_dim=256, input_length=)(merge)
  lstm = LSTM(10, return_sequences=True)(embed)
  # interpretation model
  hidden1 = Dense(10, activation='relu')(lstm)
  output = Dense(1, activation='sigmoid')(hidden1)
  model = Model(inputs=[visible1, visible2], outputs=output)
  # summarize layers
  print(model.summary())
  plot_model(model, show_shapes=True, dpi=64)
  

def cascade_model():
    window_size = 5
    mesh_rows = 20
    mesh_columns = 22
    cnn_activation ="tanh"
    lstm_activation="tanh"
    model_activation="sigmoid"
    pool_size = (1,1)
    number_conv2D_filters = 10
    kernel_shape = (7)
    embedding_output_dim = 256
    number_lstm_cells = 10
    number_nodes_hidden = 100
    inputs = []
    convs = []
    
    for i in range(window_size):
        input = Input(shape=(mesh_rows, mesh_columns,1), name = "input"+str(i+1))
        inputs.append(input)
    
    for i in range(window_size):
        conv = Conv2D(number_conv2D_filters, kernel_shape, activation=cnn_activation, input_shape=(mesh_rows, mesh_columns,1))(inputs[i])# modify shape and kernel
        pool = MaxPool2D(pool_size=pool_size)(conv) # modify pool size
        convs.append(Flatten()(pool))
    
    merge = concatenate(convs)
    #embed = Embedding(input_dim=11200, output_dim=embedding_output_dim, input_length=11200)(merge)#hard coded value of 11200 = 5 * 2240 but we can see later how to get it programatically
    #hid = Dense(100,activation=model_activation)(merge)
    merge = Reshape(target_shape=(1,11200,))(merge)
    lstm = LSTM(number_lstm_cells, return_sequences=False)(merge)
    hidden1 = Dense(number_nodes_hidden, activation=lstm_activation)(lstm)
    output = Dense(1, activation=model_activation,name="out")(hidden1)
    
    model = Model(inputs=inputs, outputs=output)
    
#    print(model.summary())
#    plot_model(model, show_shapes=True, dpi=50)
    return model

model = cascade_model()


prop = optimizers.RMSprop(1e-3)
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

mesh_rows = 20
mesh_columns = 22
X = np.random.rand(1000,mesh_rows,mesh_columns)
x_val = np.random.rand(50,mesh_rows,mesh_columns)
#X = np.random.randint(0,10,(1000,mesh_rows,mesh_columns))
number_y_examples = int(X.shape[0]/5)
Y = np.random.randint(0,2,(number_y_examples,1))
y_val = np.random.randint(0,2,(10,1))
print(Y.dtype)
Y = Y.astype(np.float)
y_val =y_val.astype(np.float)

#print(Y.shape)
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#print(X_train.shape)
optimizer = "sgd"
loss = "binary_crossentropy"
#loss = "mse"
#loss= "mean_squared_error"
#loss="sparse_categorical_crossentropy"


#print(X_train.shape)
#number_training_parts = int(X_train.shape[0]/5)
#print(number_training_parts)
input1 = X[0:200]
input1 = np.reshape(input1,(200,20,22,1))
i1_val = np.reshape(x_val[0:10],(10,20,22,1))
#print(input1)
#print(input1.shape)
input2 = X[200:400]
input2 = np.reshape(input2,(200,20,22,1))
i2_val = np.reshape(x_val[10:20],(10,20,22,1))


input3 = X[400:600]
input3 = np.reshape(input3,(200,20,22,1))
i3_val = np.reshape(x_val[20:30],(10,20,22,1))

input4 = X[600:800]
input4 = np.reshape(input4,(200,20,22,1))
i4_val = np.reshape(x_val[30:40],(10,20,22,1))

input5 = X[800:]
input5 = np.reshape(input5,(200,20,22,1))
i5_val = np.reshape(x_val[40:],(10,20,22,1))





test1 = np.reshape(input1[0],(1,20,22,1))
test2 = np.reshape(input2[0],(1,20,22,1))
test3 = np.reshape(input3[0],(1,20,22,1))
test4 = np.reshape(input4[0],(1,20,22,1))
test5 = np.reshape(input5[0],(1,20,22,1))

input_dictionary = {'input1' : input1,'input2' : input2,'input3' : input3, 'input4': input4,'input5':input5}
input_val_dict = {'input1' : i1_val,'input2' : i2_val,'input3' : i3_val, 'input4': i4_val,'input5':i5_val}
#input_dictionary = {'input1' : test1,'input2' : test2,'input3' : test3, 'input4': test4,'input5':test5}
model.compile(optimizer=prop, loss=loss, metrics=["accuracy"])
print(input4[0].shape)
y_train = Y[0:200]
#y_train = Y[0]
#print(y_train)
output_dict = {"out":y_train}
output_val_dict = {"out":y_val}

model.fit(input_dictionary,output_dict,validation_data=(input_val_dict,output_val_dict), batch_size=100,epochs=400)
for i in range (2):
    
    validate1 = np.reshape(input1[i],(1,20,22,1))
    validate2 = np.reshape(input2[i],(1,20,22,1))
    validate3 = np.reshape(input3[i],(1,20,22,1))
    validate4 = np.reshape(input4[i],(1,20,22,1))
    validate5 = np.reshape(input5[i],(1,20,22,1))

    prediction = model.predict({'input1':validate1,'input2':validate2,'input3':validate3,'input4':validate4,'input5':validate5})
    #print(input1[0].shape)
    print(prediction)
    print(prediction.shape)
    


