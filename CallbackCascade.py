# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 02:52:09 2019

@author: Smail
"""


import numpy as np
from tensorflow.keras.callbacks import Callback
import os
import re
import matplotlib.pyplot as plt



class PrintingCallback(Callback):
    def __init__(self, number_epochs, batch_size, cascade_model_object):
        self.number_epochs = number_epochs
        self.batch_size = batch_size
        self.cascade_model_object = cascade_model_object
        
    def on_train_begin(self, logs=None):
        self.experiment_number = self.get_experiment_number()
        print()
        print()
        print("-"*7 +" Beginning of Experiment {} ".format(self.experiment_number) + "-"*7)
        print()
        print()
        self.create_experiment_folder(self.experiment_number)
        self.create_summary_file(self.experiment_number)
        self.append_to_summary_file(self.cascade_model_object, self.experiment_number)
        self.create_info_epochs_file(self.experiment_number)

    def on_train_end(self, logs=None):
        print()
        print()
        print("-"*7 +" End of Experiment {} ".format(self.experiment_number) + "-"*7)
        print()
        print()
        print("-"*7 +" Plotting and saving the epochs training/validation accuracy/loss " + "-"*7)
        self.plot_epochs_info(self.experiment_number)

    def on_epoch_end(self, epoch, logs=None):
      self.append_to_epochs_file(self.experiment_number,epoch, logs['accuracy'], logs['loss'], logs['val_accuracy'], logs['val_loss'])

      
    #Assumes an existing folder called Experiments
    #Checks if the folder is empty or not
    #If empty, no experiment was done and return 1 (meaning 1st experiment)
    #IF not empty, check all folders, extract max, and return max + 1
    def get_experiment_number(self):
        experiments_folders_list = os.listdir(path='Experiments')
        if(len(experiments_folders_list) == 0): #empty folder
            return 1
        else:  
            temp_numbers=[]
            for folder in experiments_folders_list:
                number = re.findall(r'\d+', folder)
                temp_numbers.append(int(number[0]))
            return max(temp_numbers) + 1
        
        
    def create_experiment_folder(self,experiment_number):
        try:
            path_new_experiment = "Experiments/Experiment" + str(experiment_number)
            os.mkdir(path_new_experiment)
        except Exception as e:
            print ("Creation of the directory %s failed" % path_new_experiment)
            print("Exception error: ",str(e))      
      
    def on_predict_batch_end(self, batch, logs=None):
        print("TO DO : write prediction in info_prediction.txt")
        
    def create_summary_file(self,experiment_number):
        filename = "Experiments/Experiment"+str(experiment_number)+"/summary_model"+str(experiment_number)+".txt"
        with open(filename, "w") as file:
            file.write("Summary of the model used for experiment"+str(experiment_number)+" : \n\n")
        
    def append_to_summary_file(self, model_object, experiment_number):
        filename = "Experiments/Experiment"+str(experiment_number)+"/summary_model"+str(experiment_number)+".txt"
        with open(filename, "a+") as file:
            file.write("window_size: "+str(model_object.window_size)+"\n")#.format(str(model_object.window_size)))
            file.write("cnn_activation: {}\n".format(str(model_object.cnn_activation)))
            file.write("hidden_activation: {}\n".format(str(model_object.hidden_activation)))
            file.write("model_activation: {}\n".format(str(model_object.model_activation)))
            file.write("pool_size: {}\n".format(str(model_object.pool_size)))
            file.write("number_conv2D_filters: {}\n".format(str(model_object.number_conv2D_filters)))
            file.write("kernel_shape: {}\n".format(str(model_object.kernel_shape)))
            file.write("number_lstm_cells: {}\n".format(str(model_object.number_lstm_cells)))
            file.write("number_nodes_hidden: {}\n".format(str(model_object.number_nodes_hidden)))
            file.write("loss: {}\n".format(str(model_object.loss)))
            file.write("optimizer: {}\n".format(str(model_object.optimizer)))
            
    def create_info_epochs_file(self,experiment_number):
        filename = "Experiments/Experiment"+str(experiment_number)+"/info_epochs_model"+str(experiment_number)+".txt"
        with open(filename, "w") as file:
            file.write("")
        
    def append_to_epochs_file(self,experiment_number, epoch_number, training_accuracy, training_loss, validation_accuracy, validation_loss):
        filename = "Experiments/Experiment"+str(experiment_number)+"/info_epochs_model"+str(experiment_number)+".txt"
        with open(filename, "a+") as file:
            file.write("Epoch {0},training_acuracy:{1:.2f},trainig_loss:{2:.2f},validation_accuracy:{3:.2f},validation_loss:{4:.2f}\n".format(epoch_number,training_accuracy,training_loss,validation_accuracy,validation_loss))
        
    def create_info_epochs_prediction(self, experiment_number):
        filename = "Experiments/Experiment"+str(experiment_number)+"/info_prediction_model"+str(experiment_number)+".txt"
        with open(filename, "w") as file:
            file.write("Prediction of the model used for experiment"+str(experiment_number)+" : \n\n")
            
    def plot_epochs_info(self,experiment_number):
        filename = "Experiments/Experiment"+str(experiment_number)+"/info_epochs_model"+str(experiment_number)+".txt"
        train_accuracies = []
        train_losses = []
        valid_accuracies = []
        valid_losses = []
        try:
            with open(filename, "r") as file:
                lines = file.readlines()
                number_epochs = len(lines)
                x_values = np.arange(start = 1, stop = number_epochs + 1)
                for line in lines:
                    temp_parts = line.split(',')
                    
                    train_accuracy_part = temp_parts[1]
                    train_accuracies.append(float(train_accuracy_part.split(':')[1]))
                    
                    train_loss_part = temp_parts[2]
                    train_losses.append(float(train_loss_part.split(':')[1]))
                    
                    valid_accuracy_part = temp_parts[3]
                    valid_accuracies.append(float(valid_accuracy_part.split(':')[1]))
                    
                    valid_loss_part = temp_parts[4]
                    valid_losses.append(float(valid_loss_part.split(':')[1]))
                    
        except Exception as e:
            print("Problem while reading the file {}".format(filename))
            print("Exception message : {}".format(e))
        plt.figure(figsize=(10,10))
        plt.plot(x_values,train_accuracies,label="Training accuracy")
        plt.plot(x_values,train_losses,label="Training loss")
        plt.plot(x_values,valid_accuracies,label="Validation accuracy")
        plt.plot(x_values,valid_losses,label="Validation loss")
        plt.legend(fontsize='small')
        plt.title("Accuracy and Loss during Training and Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy and Loss")
        output_filename = "Experiments/Experiment"+str(experiment_number)+"/plot_model"+str(experiment_number)+".jpg"
        plt.savefig(output_filename)
        plt.show()

        
#Todo 04/01 : change colors of curves
        #make 2 separate plots , one for accuracies , other for loss
        #change destination folder to make experiments specific to cascade model
        #on_predict_batch_end
        

