# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 02:52:09 2019

@author: Smail
"""
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers
import numpy as np
import CascadeModel as cascade
from tensorflow.keras.callbacks import Callback

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
    
def prepare_data():
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
    
    
    model = cascade.cascade_model()
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    y_train = Y[0:200]
    return input_dictionary,y_train
    

"""# Training with synthetic data"""

def model_training(model, input_training, output_training,batch_size,epochs):
    model.fit(input_training,output_training,batch_size=batch_size,epochs=epochs)
    
def model_training_callbacks(model, input_training, output_training,batch_size,epochs,callbacks):
    model.fit(input_training,output_training,batch_size=batch_size,epochs=epochs)
    
    
class PrintingCallback(Callback):
    def __init__(self, number_epochs, cascade_model_object):
        self.number_epochs = number_epochs
        self.cascade_model_object = cascade_model_object
        
    def on_train_begin(self, logs=None):
        print("go in experiments folder, check if empty")
        print("if empty, create experiment0 and set self.expnumber=0")
        print("if not, fetch last experiment number, and create new exp folder")
        print("set self.expnumber = X")
        print("create a text file called 'summary_model.txt' in this new folder")
        print("write in it all fields of class")
        print("create another text file called 'info_epochs.txt'")
        
#    def on_train_batch_end(self, batch, logs=None):
#      print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))
#
#    def on_test_batch_end(self, batch, logs=None):
#      print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))

    def on_epoch_end(self, epoch, logs=None):
      print('The average loss for epoch {} is {:7.2f} and mean absolute error is {:7.2f}.'.format(epoch, logs['loss'], logs['mae']))
      print("append on 'info_epochs_training.txt' the training/validation loss and accuracy")
      
    def on_predict_batch_end(self, batch, logs=None):
        print("info_prediction.txt")
        
#Todo : finish this class
#Reading/Writing from csv
#Test this file with synthetic data
      
      