from ModelCascadeMulti import CascadeMulti
from os import listdir
from os.path import isfile, join
import numpy as np
import data_utils_multiclass as utils
import gc
from sklearn.utils import shuffle
import tensorflow as tf
import h5py
from scipy import stats
import time
import os
import re
import matplotlib.pyplot as plt
import matplotlib as mpl


def normalize_zscore(matrix):
    return stats.zscore(matrix)


def get_lists_indexes(matrix_length):
        indexes_1 = np.arange(start=0,stop = matrix_length-4,step=5)
        indexes_2 = np.arange(start=1,stop = matrix_length-3,step=5)
        indexes_3 = np.arange(start=2,stop = matrix_length-2,step=5)
        indexes_4 = np.arange(start=3,stop = matrix_length-1,step=5)
        indexes_5 = np.arange(start=4,stop = matrix_length-0,step=5)
        return (indexes_1,indexes_2,indexes_3,indexes_4,indexes_5)
    
def get_input_lists(matrix,indexes):
    inputs = []
    for i in range(5):
        inputs.append(np.take(matrix,indexes[i],axis=0))
    del matrix
    return inputs[0],inputs[1],inputs[2],inputs[3],inputs[4]

def closestNumber(n, m) : 
    q = int(n / m) 
    n1 = m * q 
    if((n * m) > 0) : 
        n2 = (m * (q + 1))  
    else : 
        n2 = (m * (q - 1)) 
    if (abs(n - n1) < abs(n - n2)) : 
        return n1 
    return n2 



def normalize_matrix(matrix):
    max,min = matrix.max(),matrix.min()
    return (matrix-min)/(max-min)

def get_dataset_name(file_name_with_dir):
    filename_without_dir = file_name_with_dir.split('/')[-1]
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name


def preprocess_data_type(matrix):
    # matrix = matrix * 1e11
    # matrix = matrix * 1e21
    matrix = normalize_matrix(matrix)

    if(matrix.shape[1] == 1):
        length = 1
    else:
        length = closestNumber(int(matrix.shape[1]) - 5,5)
        
    meshes = np.zeros((length,20,22),dtype=np.float64)
    for i in range(length):
        array_time_step = np.reshape(matrix[:,i],(1,248))
        meshes[i] = utils.array_to_mesh(array_time_step)

    del matrix

    input1,input2,input3,input4,input5 = get_input_lists(meshes, get_lists_indexes(length))

    del meshes

    number_y_labels = int(length/5)
    y_rest = np.ones((number_y_labels,1),dtype=np.int8)
    return (input1,input2,input3,input4,input5), y_rest

def load_data(file_dirs,index_batch_files):
    print("Batch number : {}".format(index_batch_files + 1))

    rest_matrix = np.random.rand(248,1)
    math_matrix = np.random.rand(248,1)
    memory_matrix = np.random.rand(248,1)
    motor_matrix = np.random.rand(248,1)
    number_classes = 4
    number_files_per_batch = 4

    files_to_load = file_dirs[index_batch_files * number_files_per_batch: (index_batch_files + 1) * number_files_per_batch]
    print()
    print("Files to load :")
    print(files_to_load)
    print()
    for i in range(len(files_to_load)):
        if "rest" in files_to_load[i]:
            with h5py.File(files_to_load[i],'r') as f:
                dataset_name = get_dataset_name(files_to_load[i])
                matrix = f.get(dataset_name)
                matrix = np.array(matrix)
            assert matrix.shape[0] == 248 , "This rest data does not have 248 channels, but {} instead".format(matrix.shape[0])
            rest_matrix = np.column_stack((rest_matrix, matrix))

        if "math" in files_to_load[i]:
            with h5py.File(files_to_load[i],'r') as f:
                dataset_name = get_dataset_name(files_to_load[i])
                matrix = f.get(dataset_name)
                matrix = np.array(matrix)
            assert matrix.shape[0] == 248 , "This math data does not have 248 channels, but {} instead".format(matrix.shape[0])
            math_matrix = np.column_stack((math_matrix, matrix))
            
        if "memory" in files_to_load[i]:
            with h5py.File(files_to_load[i],'r') as f:
                dataset_name = get_dataset_name(files_to_load[i])
                matrix = f.get(dataset_name)
                matrix = np.array(matrix)
            assert matrix.shape[0] == 248 , "This memory data does not have 248 channels, but {} instead".format(matrix.shape[0])
            memory_matrix = np.column_stack((memory_matrix, matrix))
            
        if "motor" in files_to_load[i]:
            with h5py.File(files_to_load[i],'r') as f:
                dataset_name = get_dataset_name(files_to_load[i])
                matrix = f.get(dataset_name)
                matrix = np.array(matrix)
            assert matrix.shape[0] == 248 , "This motor data does not have 248 channels, but {} instead".format(matrix.shape[0])
            motor_matrix = np.column_stack((motor_matrix, matrix))


    x_rest,y_rest = preprocess_data_type(rest_matrix)    

    y_rest = y_rest*0
    input1_rest,input2_rest,input3_rest,input4_rest,input5_rest = x_rest[0],x_rest[1],x_rest[2],x_rest[3],x_rest[4]
    gc.collect()

    x_math,y_math = preprocess_data_type(math_matrix)   
    input1_math,input2_math,input3_math,input4_math,input5_math = x_math[0],x_math[1],x_math[2],x_math[3],x_math[4]  

    gc.collect()      


    x_mem,y_mem = preprocess_data_type(memory_matrix)
    y_mem = y_mem * 2
    input1_mem,input2_mem,input3_mem,input4_mem,input5_mem = x_mem[0],x_mem[1],x_mem[2],x_mem[3],x_mem[4] 

    gc.collect()

    x_motor,y_motor = preprocess_data_type(motor_matrix)
    y_motor = y_motor * 3
    input1_motor,input2_motor,input3_motor,input4_motor,input5_motor = x_motor[0],x_motor[1],x_motor[2],x_motor[3],x_motor[4] 

    gc.collect()

    dict1 = {0:input1_rest,1:input1_math,2:input1_mem,3:input1_motor}
    dict2 = {0:input2_rest,1:input2_math,2:input2_mem,3:input2_motor}
    dict3 = {0:input3_rest,1:input3_math,2:input3_mem,3:input3_motor}
    dict4 = {0:input4_rest,1:input4_math,2:input4_mem,3:input4_motor}
    dict5 = {0:input5_rest,1:input5_math,2:input5_mem,3:input5_motor}

    input1 = np.random.rand(1,20,22)
    input2 = np.random.rand(1,20,22)
    input3 = np.random.rand(1,20,22)
    input4 = np.random.rand(1,20,22)
    input5 = np.random.rand(1,20,22)

    for i in range(number_classes):
        if dict1[i].shape[0]>0:
            input1=np.concatenate((input1,dict1[i]))
            dict1[i] = None
            
        if dict2[i].shape[0]>0:
            input2=np.concatenate((input2,dict2[i]))
            dict2[i] = None
            
        if dict3[i].shape[0]>0:
            input3=np.concatenate((input3,dict3[i]))
            dict3[i] = None
            
        if dict4[i].shape[0]>0:
            input4=np.concatenate((input4,dict4[i]))
            dict4[i] = None
            
        if dict5[i].shape[0]>0:
            input5=np.concatenate((input5,dict5[i]))
            dict5[i] = None


    #deleting first element of each lists                        
    input1 = np.delete(input1,0,0)
    input2 = np.delete(input2,0,0)
    input3 = np.delete(input3,0,0)
    input4 = np.delete(input4,0,0)
    input5 = np.delete(input5,0,0)
    
    # print("part reshaping")
    input1 = np.reshape(input1,(input1.shape[0],20,22,1))
    input2 = np.reshape(input2,(input2.shape[0],20,22,1))
    input3 = np.reshape(input3,(input3.shape[0],20,22,1))
    input4 = np.reshape(input4,(input4.shape[0],20,22,1))
    input5 = np.reshape(input5,(input5.shape[0],20,22,1))
    

    gc.collect()
    
    dict_y = {0:y_rest,1:y_math,2:y_mem,3:y_motor}
    
    y = np.random.rand(1,1)
    for i in range(number_classes):
        if dict_y[i].shape[0]>0:
            y = np.concatenate((y,dict_y[i]))

    y = np.delete(y,0,0)

    # print("part shuffling")

    input1,input2,input3,input4,input5,y = shuffle(input1,input2,input3,input4,input5, y, random_state=42)
    
    x_length = input1.shape[0]
    for i in range(x_length):
        temp1 = input1[i]
        inside1 = temp1[:,:,0]
        norm1 = normalize_matrix(inside1)
        input1[i][:,:,0] = norm1
        
        temp2 = input2[i]
        inside2 = temp2[:,:,0]
        norm2 = normalize_matrix(inside2)
        input2[i][:,:,0] = norm2
        
        temp3 = input3[i]
        inside3 = temp3[:,:,0]
        norm3 = normalize_matrix(inside3)
        input3[i][:,:,0] = norm3
        
        temp4 = input4[i]
        inside4 = temp4[:,:,0]
        norm4 = normalize_matrix(inside4)
        input4[i][:,:,0] = norm4
        
        temp5 = input5[i]
        inside5 = temp5[:,:,0]
        norm5 = normalize_matrix(inside5)
        input5[i][:,:,0] = norm5

    gc.collect()

    input1 = tf.cast(input1,tf.float32)
    input2 = tf.cast(input2,tf.float32)
    input3 = tf.cast(input3,tf.float32)
    input4 = tf.cast(input4,tf.float32)
    input5 = tf.cast(input5,tf.float32)

    data_dict = {'input1' : input1,'input2' : input2,'input3' : input3, 'input4': input4,'input5':input5}


    gc.collect()
    
    y = tf.keras.utils.to_categorical(y,number_classes)
    return data_dict,y



window_size = 5
cnn_activation ="relu"
hidden_activation="relu"
model_activation="softmax"
pool_size = (1,1)
number_conv2D_filters = 10
kernel_shape = (7)
number_lstm_cells = 10
number_nodes_hidden = 10
loss = "categorical_crossentropy"
optimizer = "sgd"
using_gpu = False


model_object = CascadeMulti(window_size,cnn_activation, hidden_activation, model_activation, pool_size,
                            number_conv2D_filters, kernel_shape, number_lstm_cells, number_nodes_hidden, 
                            loss, optimizer,using_gpu)

cascade_model = model_object.model

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def loss(model, x, y):
  y_ = model(x)

  return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

def separate_list(all_files_list):
    rest_list = []
    mem_list = []
    math_list = []
    motor_list = []
    for item in all_files_list:
        if "rest" in item:
            rest_list.append(item)
        if "memory" in item:
            mem_list.append(item)
        if "math" in item:
            math_list.append(item)
        if "motor" in item:
            motor_list.append(item)            
    return rest_list, mem_list, math_list, motor_list

def orderer_shuffling(rest_list,mem_list,math_list,motor_list):
    ordered_list = []
    for index, (value1, value2, value3, value4) in enumerate(zip(rest_list, mem_list, math_list, motor_list)):
        ordered_list.append(value1)
        ordered_list.append(value2)
        ordered_list.append(value3)
        ordered_list.append(value4)

    return ordered_list


training_file_dir = "Data/train"
all_train_files = [f for f in listdir(training_file_dir) if isfile(join(training_file_dir, f))]
train_files_dirs = []
for i in range(len(all_train_files)):
    train_files_dirs.append(training_file_dir+'/'+all_train_files[i])
rest_list, mem_list, math_list, motor_list = separate_list(train_files_dirs)
train_files_dirs = orderer_shuffling(rest_list, mem_list, math_list, motor_list)


validation_file_dir = "Data/validate"
all_validate_files = [f for f in listdir(validation_file_dir) if isfile(join(validation_file_dir, f))]
validate_files_dirs = []
for i in range(len(all_validate_files)):
    validate_files_dirs.append(validation_file_dir+'/'+all_validate_files[i])
rest_list, mem_list, math_list, motor_list = separate_list(validate_files_dirs)
validate_files_dirs = orderer_shuffling(rest_list, mem_list, math_list, motor_list)


def plot_epochs_info(experiment_number):
    filename = "Experiments/Cascade/Experiment"+str(experiment_number)+"/info_epochs_model"+str(experiment_number)+".txt"
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
    # plt.figure(figsize=(10,10))
    mpl.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(10,10))
    # plt.figure()
    ax1.plot(x_values,train_accuracies,label="Training Accuracy",color="#4C72B0")
    ax1.plot(x_values,valid_accuracies,label="Validation Accuracy", color='#55A868')
    ax1.legend(loc="upper left",fontsize='small')
    
    ax2.plot(x_values,train_losses,label="Training Loss", color = "#DD8452" )
    ax2.plot(x_values,valid_losses,label="Validation Loss", color="#C44E52")
    ax2.legend(loc="upper right",fontsize='small')
    
    ax1.set_title("Accuracy during Training and Validation")
    ax2.set_title("Loss during Training and Validation")
    plt.xlabel("Epochs")
    ax1.set(ylabel ="Accuracy")
    ax2.set(ylabel="Loss")
    output_filename = "Experiments/Cascade/Experiment"+str(experiment_number)+"/plot_model"+str(experiment_number)+".jpg"
    fig.savefig(output_filename,dpi=100)
    plt.show()

def get_experiment_number():
    experiments_folders_list = os.listdir(path='Experiments/Cascade')
    if(len(experiments_folders_list) == 0): #empty folder
        return 1
    else:  
        temp_numbers=[]
        for folder in experiments_folders_list:
            number = re.findall(r'\d+', folder)
            if(len(number)>0):
                temp_numbers.append(int(number[0]))
        return max(temp_numbers) + 1

def append_to_epochs_file(experiment_number, epoch_number, training_accuracy, training_loss, validation_accuracy, validation_loss):
    filename = "Experiments/Cascade/Experiment"+str(experiment_number)+"/info_epochs_model"+str(experiment_number)+".txt"
    with open(filename, "a+") as file:
        file.write("Epoch {0},training_acuracy:{1:.2f},trainig_loss:{2:.2f},validation_accuracy:{3:.2f},validation_loss:{4:.2f}\n".format(epoch_number,training_accuracy,training_loss,validation_accuracy,validation_loss))

def create_cascade_folder():
    path_cascade = "Experiments/Cascade"
    if(not os.path.isdir(path_cascade)):
        try:
            os.mkdir(path_cascade)
        except Exception as e:
            print ("Creation of the main cascade experiment directory failed")
            print("Exception error: ",str(e))     
        
def create_experiment_folder(experiment_number):
    try:
        path_new_experiment = "Experiments/Cascade/Experiment" + str(experiment_number)
        check_point_path = path_new_experiment+"/checkpoints"
        os.mkdir(path_new_experiment)
        os.mkdir(check_point_path)
    except Exception as e:
        print ("Creation of the directory {} or {} failed".format(path_new_experiment,check_point_path))
        print("Exception error: ",str(e))  


def create_main_experiment_folder():
    if(not os.path.isdir("Experiments")):
        try:
            os.mkdir("Experiments")
        except Exception as e:
            print ("Creation of the main experiment directory failed")
            print("Exception error: ",str(e))


def create_info_epochs_file(experiment_number):
        filename = "Experiments/Cascade/Experiment"+str(experiment_number)+"/info_epochs_model"+str(experiment_number)+".txt"
        with open(filename, "w") as file:
            file.write("")
            
def create_summary_file(self,experiment_number):
        filename = "Experiments/Cascade/Experiment"+str(experiment_number)+"/summary_model"+str(experiment_number)+".txt"
        with open(filename, "w") as file:
            file.write("Summary of the model used for experiment"+str(experiment_number)+" : \n\n")

def on_train_begin(using_gpu,cascade_model_object):
    create_main_experiment_folder()
    create_cascade_folder()
    experiment_number = get_experiment_number()
    create_experiment_folder(experiment_number)
    print()
    print()
    if(using_gpu):
        print("-"*7 +" Beginning of Experiment {} of the Cascade model using GPU ".format(experiment_number) + "-"*7)
    else:
        print("-"*7 +" Beginning of Experiment {} of the Cascade model without using GPU".format(experiment_number) + "-"*7)            
    print()
    print()
    # self.create_experiment_folder(self.experiment_number)
    create_summary_file(experiment_number)
    append_to_summary_file(cascade_model_object, experiment_number)
    create_info_epochs_file(experiment_number)
    return experiment_number

def on_train_end(experiment_number):
    print()
    print()
    print("-"*7 +" End of Experiment {} ".format(experiment_number) + "-"*7)
    print()
    print()
    print("-"*7 +" Plotting and saving the epochs training/validation accuracy/loss " + "-"*7)
    plot_epochs_info(experiment_number)

def on_epoch_end(epoch, accuracy, loss, val_accuracy, val_loss,experiment_number,model):
    try:
        append_to_epochs_file(experiment_number,epoch, accuracy, loss, val_accuracy, val_loss)
        model_checkpoint(experiment_number, model, val_accuracy, epoch)
    except Exception as e:
        print("Failed to append in epoch file or to save the weights ...")
        print("Exception error: ",str(e))

def model_checkpoint(experiment_number,model,validation_accuracy,epoch):
    exp_path = "Experiments/Cascade/Experiment" + str(experiment_number)
    check_point_path = exp_path+"/checkpoints" + '/checkpoint-epoch_{:03d}-val_acc_{:.3f}.hdf5'.format(epoch,validation_accuracy)
    model.save_weights(check_point_path)

def append_to_summary_file(self, model_object, experiment_number):
        filename = "Experiments/Cascade/Experiment"+str(experiment_number)+"/summary_model"+str(experiment_number)+".txt"
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




# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 20
batch_size = 128
epoch_loss_avg = tf.keras.metrics.Mean()
epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()


# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 20
batch_size = 128
epoch_loss_avg = tf.keras.metrics.Mean()
epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()


def custom_training(cascade_model_object):
    accuracies_temp_train = []
    losses_temp_train= []

    accuracies_train = []#per epoch
    losses_train = []#per epoch

    accuracies_temp_val = []
    losses_temp_val = []

    accuracies_val = []#per epoch
    losses_val = []#per epoch

    with tf.device('/device:GPU:0'):
        start_time = time.time()
        number_files_per_batch = 4
        experiment_number = on_train_begin(True,cascade_model_object)
        ### Training part ###
        print("Training :")
        total_number_train_files = len(train_files_dirs)
        number_batches_training = total_number_train_files // number_files_per_batch
        for epoch in range(num_epochs):
            index_batch_train = 0
            x_batch_train, y_batch_train = load_data(train_files_dirs, index_batch_train)
            steps_per_batch_training = y_batch_train.shape[0] // batch_size
            
            length_samples = y_batch_train.shape[0]
            print("sample training length : {}".format(length_samples))
            print("number of training batches should be : {}".format(number_batches_training))
            for i in range(number_batches_training):
                for j in range(steps_per_batch_training):
                    input1 = np.reshape(x_batch_train['input1'][j*batch_size:(j+1)*batch_size],(batch_size,20,22,1))
                    input2 = np.reshape(x_batch_train['input2'][j*batch_size:(j+1)*batch_size],(batch_size,20,22,1))
                    input3 = np.reshape(x_batch_train['input3'][j*batch_size:(j+1)*batch_size],(batch_size,20,22,1))
                    input4 = np.reshape(x_batch_train['input4'][j*batch_size:(j+1)*batch_size],(batch_size,20,22,1))
                    input5 = np.reshape(x_batch_train['input5'][j*batch_size:(j+1)*batch_size],(batch_size,20,22,1))
                    x_step_dict = {'input1' : input1,'input2' : input2,'input3' : input3, 'input4': input4,'input5':input5}
                    y_step = y_batch_train[j*batch_size:(j+1)*batch_size]

                    loss_value, grads = grad(cascade_model, x_step_dict, y_step)
                    optimizer.apply_gradients(zip(grads, cascade_model.trainable_variables))

                    # # Track progress
                    epoch_loss_avg(loss_value)  # Add current batch loss
                    # # Compare predicted label to actual label
                    epoch_accuracy(y_step, cascade_model(x_step_dict)) # batch accuracy

                print("Training Accuracy: {:.3%}".format(epoch_accuracy.result()))
                print("Loss Training: {:.3f}".format(epoch_loss_avg.result()))
                accuracies_temp_train.append(epoch_accuracy.result())
                losses_temp_train.append(loss_value)
                index_batch_train += 1
                x_batch_train, y_batch_train = load_data(train_files_dirs, index_batch_train)
                steps_per_batch_training = y_batch_train.shape[0] // batch_size


            ### Validation Part ###
            print("Validation :")
            total_number_validate_files = len(validate_files_dirs)
            number_batches_validation = total_number_validate_files // number_files_per_batch            
            index_batch_validate = 0
            x_batch_validate, y_batch_validate = load_data(validate_files_dirs, index_batch_validate)
            steps_per_batch_validation = y_batch_validate.shape[0] // batch_size
            for k in range(number_batches_validation):
                for l in range(steps_per_batch_validation):
                    input1_val = np.reshape(x_batch_validate['input1'][j*batch_size:(j+1)*batch_size],(batch_size,20,22,1))
                    input2_val = np.reshape(x_batch_validate['input2'][j*batch_size:(j+1)*batch_size],(batch_size,20,22,1))
                    input3_val = np.reshape(x_batch_validate['input3'][j*batch_size:(j+1)*batch_size],(batch_size,20,22,1))
                    input4_val = np.reshape(x_batch_validate['input4'][j*batch_size:(j+1)*batch_size],(batch_size,20,22,1))
                    input5_val = np.reshape(x_batch_validate['input5'][j*batch_size:(j+1)*batch_size],(batch_size,20,22,1))
                    x_step_dict = {'input1' : input1_val,'input2' : input2_val,'input3' : input3_val, 'input4': input4_val,'input5':input5_val}
                    y_step = y_batch_validate[j*batch_size:(j+1)*batch_size]

                    loss_value, grads = grad(cascade_model, x_step_dict, y_step)
                    optimizer.apply_gradients(zip(grads, cascade_model.trainable_variables))

                    epoch_loss_avg(loss_value)  # Add current batch loss
                    epoch_accuracy(y_step, cascade_model(x_step_dict)) # batch accuracy

                print("Validation Accuracy: {:.3%}".format(epoch_accuracy.result()))
                print("Loss Validation: {:.3f}".format(epoch_loss_avg.result()))
                accuracies_temp_val.append(epoch_accuracy.result())
                losses_temp_val.append(loss_value)
                index_batch_validate += 1
                x_batch_validate, y_batch_validate = load_data(validate_files_dirs, index_batch_validate)
                steps_per_batch_validation = y_batch_validate.shape[0] // batch_size
                    


            # End epoch
            print("Epoch {:03d}".format(epoch))

            ## Training Information ##
            average_loss_epoch_train = sum(losses_temp_train)/len(losses_temp_train)
            print("Epoch Training Loss : {:.3f}".format(average_loss_epoch_train))
            losses_train.append(average_loss_epoch_train)
            losses_temp_train = []

            average_accuracy_epoch_train = sum(accuracies_temp_train)/len(accuracies_temp_train)
            print("Epoch Training Accuracy: {:.3%}".format(average_accuracy_epoch_train))
            accuracies_train.append(average_accuracy_epoch_train)
            accuracies_temp_train = []

            ## Validation Information ##
            average_loss_epoch_validate = sum(losses_temp_val)/len(losses_temp_val)
            print("Epoch Validation Loss : {:.3f}".format(average_loss_epoch_validate))
            losses_val.append(average_loss_epoch_validate)
            losses_temp_val = []

            average_accuracy_epoch_validate = sum(accuracies_temp_val)/len(accuracies_temp_val)
            print("Epoch Validation Accuracy: {:.3%}".format(average_accuracy_epoch_validate))
            accuracies_val.append(average_accuracy_epoch_validate)
            accuracies_temp_val = []

            on_epoch_end(epoch, average_accuracy_epoch_train, average_loss_epoch_train, \
                         average_accuracy_epoch_validate, average_loss_epoch_validate, experiment_number, cascade_model)


       
    stop_time = time.time()
    time_span = stop_time - start_time
    print()
    print()
    print("training took {:.2f} seconds".format(time_span))
    on_train_end(experiment_number)


custom_training(model_object)      


        


        