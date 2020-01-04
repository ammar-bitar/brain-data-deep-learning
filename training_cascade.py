# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 16:49:23 2020

@author: Smail
"""
from tensorflow.keras.utils import plot_model
import numpy as np
import CascadeModel as cascade
from tensorflow.keras import optimizers
from CallbackCascade import PrintingCallback

window_size = 5
cnn_activation ="tanh"
hidden_activation="tanh"
model_activation="sigmoid"
pool_size = (1,1)
number_conv2D_filters = 10
kernel_shape = (7)
number_lstm_cells = 10
number_nodes_hidden = 10

optimizer = "sgd"
sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
prop = optimizers.RMSprop(1e-3)

loss = "binary_crossentropy"


def show_info_model():
    model = cascade.Cascade(window_size,cnn_activation, hidden_activation, model_activation, pool_size,
                            number_conv2D_filters, kernel_shape, number_lstm_cells, number_nodes_hidden, 
                            loss, optimizer)
    print(model.summary())
    plot_model(model, show_shapes=True, dpi=100)


def prepare_synthetic_data():
    mesh_rows = 20
    mesh_columns = 22
    X = np.random.randint(0,10,(1000,mesh_rows,mesh_columns))
    number_y_examples = int(X.shape[0]/5)
    Y = np.random.randint(0,2,(number_y_examples,1))

    
    input1 = X[0:200]
    input1 = np.reshape(input1,(200,20,22,1))
    
    input2 = X[200:400]
    input2 = np.reshape(input2,(200,20,22,1))
    
    input3 = X[400:600]
    input3 = np.reshape(input3,(200,20,22,1))
    
    input4 = X[600:800]
    input4 = np.reshape(input4,(200,20,22,1))
    
    input5 = X[800:]
    input5 = np.reshape(input5,(200,20,22,1))
    

    input_dictionary = {'input1' : input1,'input2' : input2,'input3' : input3, 'input4': input4,'input5':input5}
    

    y_train = Y[0:200]
    return input_dictionary,y_train


def model_training(model, input_training, output_training,batch_size,epochs):
    model.fit(input_training,output_training, validation_data=(input_training, output_training), batch_size=batch_size,epochs=epochs)
    
def model_training_callbacks(model, input_training, output_training,batch_size,epochs,callbacks):
    model.fit(input_training,output_training, validation_data=(input_training, output_training), batch_size=batch_size,epochs=epochs, callbacks = callbacks)


def test_training_synthetic_data():  
    model_object = cascade.Cascade(window_size,cnn_activation, hidden_activation, model_activation, pool_size,
                                number_conv2D_filters, kernel_shape, number_lstm_cells, number_nodes_hidden, 
                                loss, optimizer)
    cascade_model = model_object.model
    cascade_model.compile(optimizer=model_object.optimizer, loss=model_object.loss, metrics=["accuracy"])
    
    x_training, y_training = prepare_synthetic_data()
    
    batch_size = 64
    epochs = 5
    
    model_training_callbacks(cascade_model, x_training, y_training, batch_size,epochs, callbacks = [PrintingCallback(epochs, batch_size, model_object)])
    

#test_training_synthetic_data()
