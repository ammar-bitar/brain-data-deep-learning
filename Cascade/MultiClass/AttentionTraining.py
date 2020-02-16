import time
import tensorflow as tf
from AttentionModelCascadeMulti import AttentionCascadeMulti
from os import listdir
from os.path import isfile, join
import numpy as np
import data_utils_multi as utils
import gc
from scipy import stats
import os
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
from multiprocessing import Pool




def create_summary_file(experiment_number):
        filename = "Experiments/Cascade/Experiment"+str(experiment_number)+"/summary_model"+str(experiment_number)+".txt"
        with open(filename, "w") as file:
            file.write("Summary of the model used for experiment"+str(experiment_number)+" : \n\n")

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
    output_filename = "Experiments/Cascade/Experiment"+str(experiment_number)+"/plot_model"+str(experiment_number)+".png"
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

def on_train_begin(cascade_model_object):
    create_main_experiment_folder()
    create_cascade_folder()
    experiment_number = get_experiment_number()
    create_experiment_folder(experiment_number)
    print()
    print()
    print("-"*7 +" Beginning of Experiment {} of the Cascade model ".format(experiment_number) + "-"*7)            
    print()
    print()
    # self.create_experiment_folder(self.experiment_number)
    create_summary_file(experiment_number)
    append_to_summary_file(cascade_model_object, experiment_number)
    create_info_epochs_file(experiment_number)
    create_info_test_file(experiment_number)
    return experiment_number

def on_train_end(experiment_number):
    print()
    print()
    print("-"*7 +" End of Experiment {} ".format(experiment_number) + "-"*7)
    print()
    print()
    print("-"*7 +" Plotting and saving the epochs training/validation accuracy/loss " + "-"*7)
    plot_epochs_info(experiment_number)

def save_training_time(experiment_number,time):
    filename = "Experiments/Cascade/Experiment"+str(experiment_number)+"/summary_model"+str(experiment_number)+".txt"
    with open(filename, "a+") as file:
        file.write("\n\nTraining time: {:.2f} seconds".format(time))

def write_comment(experiment_number,comment):
    filename = "Experiments/Cascade/Experiment"+str(experiment_number)+"/summary_model"+str(experiment_number)+".txt"
    with open(filename, "a+") as file:
        file.write("\nAttention model, comment :  {}".format(comment))


def on_epoch_end(epoch, accuracy, loss, val_accuracy, val_loss,experiment_number,model):
    try:
        append_to_epochs_file(experiment_number,epoch, accuracy, loss, val_accuracy, val_loss)
        model_checkpoint(experiment_number, model, val_accuracy, epoch)
    except Exception as e:
        print("Failed to append in epoch file or saving weights...")
        print("Exception error: ",str(e))

def model_save(experiment_number,model):
    exp_path = "Experiments/Cascade/Experiment" + str(experiment_number)
    check_point_path = exp_path+"/model{}.h5".format(experiment_number)
    model.save(check_point_path)
    
def model_checkpoint(experiment_number,model,validation_accuracy,epoch):
    exp_path = "Experiments/Cascade/Experiment" + str(experiment_number)
    checkpoint_path = exp_path+"/checkpoints" + '/checkpoint-epoch_{:03d}-val_acc_{:.3f}.hdf5'.format(epoch,validation_accuracy)
    model.save_weights(checkpoint_path)
    

def append_to_summary_file(model_object, experiment_number):
        filename = "Experiments/Cascade/Experiment"+str(experiment_number)+"/summary_model"+str(experiment_number)+".txt"
        with open(filename, "a+") as file:
            file.write("window_size: "+str(model_object.window_size)+"\n")#.format(str(model_object.window_size)))
            file.write("conv1_filters: {}\n".format(str(model_object.conv1_filters)))
            file.write("conv2_filters: {}\n".format(str(model_object.conv2_filters)))
            file.write("conv3_filters: {}\n".format(str(model_object.conv3_filters)))
            file.write("conv1_kernel_shape: {}\n".format(str(model_object.conv1_kernel_shape)))
            file.write("conv2_kernel_shape: {}\n".format(str(model_object.conv2_kernel_shape)))
            file.write("conv3_kernel_shape: {}\n".format(str(model_object.conv3_kernel_shape)))
            file.write("padding1: {}\n".format(str(model_object.padding1)))
            file.write("padding2: {}\n".format(str(model_object.padding2)))
            file.write("padding3: {}\n".format(str(model_object.padding3)))
            file.write("conv1_activation: {}\n".format(str(model_object.conv1_activation)))
            file.write("conv2_activation: {}\n".format(str(model_object.conv2_activation)))
            file.write("conv3_activation: {}\n".format(str(model_object.conv3_activation)))
            file.write("dense_nodes: {}\n".format(str(model_object.dense_nodes)))
            file.write("dense_activation: {}\n".format(str(model_object.dense_activation)))
            file.write("dense_dropout: {}\n".format(str(model_object.dense_dropout)))
            file.write("lstm1_cells: {}\n".format(str(model_object.lstm1_cells)))
            file.write("lstm2_cells: {}\n".format(str(model_object.lstm2_cells)))
            file.write("dense3_nodes: {}\n".format(str(model_object.dense3_nodes)))
            file.write("dense3_activation: {}\n".format(str(model_object.dense3_activation)))
            file.write("final_dropout: {}\n".format(str(model_object.final_dropout)))
            
def create_info_test_file(experiment_number):
    filename = "Experiments/Cascade/Experiment"+str(experiment_number)+"/info_test_model"+str(experiment_number)+".txt"
    with open(filename, "w") as file:
        file.write("")    
        
def append_individual_test(experiment_number,epoch_number,subject,testing_accuracy):
    filename = "Experiments/Cascade/Experiment"+str(experiment_number)+"/info_test_model"+str(experiment_number)+".txt"
    with open(filename, "a+") as file:
        file.write("\nEpoch {0},test for subject '{1}',testing_accuracy:{2:.2f}".format(epoch_number,subject,testing_accuracy))

def append_average_test(experiment_number,epoch_number,testing_accuracy):
    filename = "Experiments/Cascade/Experiment"+str(experiment_number)+"/info_test_model"+str(experiment_number)+".txt"
    with open(filename, "a+") as file:
        file.write("\nEpoch {0},average_testing_accuracy:{1:.2f}\n\n".format(epoch_number,testing_accuracy))




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


def multi_processing(directory, length, num_cpu):
    assert len(directory) == length*num_cpu,"Directory does not have {} files.".format(length*num_cpu)
    window_size = 10
    split = []

    for i in range(num_cpu):
        split.append(directory[i*length:(i*length)+length])

    pool = Pool(num_cpu)
    results = pool.map(utils.load_overlapped_data, split)
    pool.terminate()
    pool.join()
    
    y = np.random.rand(1,4)
    for i in range(len(results)):
        y = np.concatenate((y,results[i][1]))

    y = np.delete(y,0,0)
    gc.collect()
    
    x={}

    x_temp = np.random.rand(1,20,21,1)
    for i in range(window_size):
        for j in range(len(results)):
            x_temp = np.concatenate((x_temp,results[j][0]["input"+str(i+1)]))
        x_temp= np.delete(x_temp,0,0)
        x["input"+str(i+1)] = x_temp
        x_temp = np.random.rand(1,20,21,1)
        gc.collect()
    return x, y

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


test_file_dir = "Data/test"
all_test_files = [f for f in listdir(test_file_dir) if isfile(join(test_file_dir, f))]
test_files_dirs = []
for i in range(len(all_test_files)):
    test_files_dirs.append(test_file_dir+'/'+all_test_files[i])
rest_list, mem_list, math_list, motor_list = separate_list(test_files_dirs)
test_files_dirs = orderer_shuffling(rest_list, mem_list, math_list, motor_list)


window_size = 10
    
conv1_filters = 4
conv2_filters = 8
conv3_filters = 16

conv1_kernel_shape = (7,7)
conv2_kernel_shape = conv1_kernel_shape
conv3_kernel_shape = conv1_kernel_shape

padding1 = "same"
padding2 = padding1
padding3 = padding1

conv1_activation = "relu"
conv2_activation = conv1_activation
conv3_activation = conv1_activation

dense_nodes = 500
dense_activation = "relu"
dense_dropout = 0.5

lstm1_cells = 10
lstm2_cells = lstm1_cells

dense3_nodes = dense_nodes
dense3_activation = "relu"
final_dropout = 0.5


cascade_attention_object = AttentionCascadeMulti(window_size,conv1_filters,conv2_filters,conv3_filters,
                conv1_kernel_shape,conv2_kernel_shape,conv3_kernel_shape,
                padding1,padding2,padding3,conv1_activation,conv2_activation,
                conv3_activation,dense_nodes,dense_activation,dense_dropout,
                lstm1_cells,lstm2_cells,dense3_nodes,dense3_activation,
                final_dropout)
cascade_model = cascade_attention_object.model



# subjects = ['106521','108323','109123']
# subjects = ['106521']
# list_subjects_test = ['212318']
train_loss_results = []
train_accuracy_results = []

num_epochs = 1
batch_size = 64






def hybrid_training(setup):
    if setup == 0:#used for quick tests
        subjects = ['105923']
        list_subjects_test = ['212318']
    if setup == 1:
        subjects = ['105923','164636','133019']
        list_subjects_test = ['204521','212318','162935','707749','725751','735148']
    if setup == 2:
        subjects = ['105923','164636','133019','113922','116726','140117']
        list_subjects_test = ['204521','212318','162935','707749','725751','735148']
    if setup == 3:
        subjects = ['105923','164636','133019','113922','116726','140117','175237','177746','185442']
        list_subjects_test = ['204521','212318','162935','707749','725751','735148']
    if setup == 4:
        subjects = ['105923','164636','133019','113922','116726','140117','175237','177746','185442','191033','191437','192641']
        list_subjects_test = ['204521','212318','162935','707749','725751','735148']
    
    subjects_string = ",".join([subject for subject in subjects])
    comment = "Training with subjects : " + subjects_string
    
    accuracies_temp_train = []
    losses_temp_train= []
    
    accuracies_train = []#per epoch
    losses_train = []#per epoch
    
    accuracies_temp_val = []
    losses_temp_val = []
    
    accuracies_val = []#per epoch
    losses_val = []#per epoch
    # with tf.device('/device:GPU:0'):
    start_time = time.time()
    #mirrored_strategy = tf.distribute.MirroredStrategy()
    #with mirrored_strategy.scope():
    cascade_model = cascade_attention_object.cascade_model()
    cascade_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
    
    experiment_number = on_train_begin(cascade_attention_object)
    for epoch in range(num_epochs):
        print("\n\n Epoch",epoch+1," \n")
        for subject in subjects:
            start_subject_time = time.time()
            print("-- Training on subject", subject)
            subject_files_train = []
            for item in train_files_dirs:
                if subject in item:
                    subject_files_train.append(item)
            
            subject_files_val = []
            for item in validate_files_dirs:
                if subject in item:
                    subject_files_val.append(item)
            number_workers_training = 16
            number_files_per_worker = len(subject_files_train)//number_workers_training
            X_train, Y_train = multi_processing(subject_files_train,number_files_per_worker,number_workers_training)
            length_training = Y_train.shape[0]
            length_adapted_batch_size= closestNumber(length_training-batch_size,batch_size)
            input1 = np.reshape(X_train['input1'][0:length_adapted_batch_size],(length_adapted_batch_size,20,21,1))
            input2 = np.reshape(X_train['input2'][0:length_adapted_batch_size],(length_adapted_batch_size,20,21,1))
            input3 = np.reshape(X_train['input3'][0:length_adapted_batch_size],(length_adapted_batch_size,20,21,1))
            input4 = np.reshape(X_train['input4'][0:length_adapted_batch_size],(length_adapted_batch_size,20,21,1))
            input5 = np.reshape(X_train['input5'][0:length_adapted_batch_size],(length_adapted_batch_size,20,21,1))
            input6 = np.reshape(X_train['input6'][0:length_adapted_batch_size],(length_adapted_batch_size,20,21,1))
            input7 = np.reshape(X_train['input7'][0:length_adapted_batch_size],(length_adapted_batch_size,20,21,1))
            input8 = np.reshape(X_train['input8'][0:length_adapted_batch_size],(length_adapted_batch_size,20,21,1))
            input9 = np.reshape(X_train['input9'][0:length_adapted_batch_size],(length_adapted_batch_size,20,21,1))
            input10 = np.reshape(X_train['input10'][0:length_adapted_batch_size],(length_adapted_batch_size,20,21,1))
            X_train = {'input1' : input1,'input2' : input2,'input3' : input3, 'input4': input4,'input5':input5,'input6' : input6,'input7' : input7,'input8' : input8, 'input9': input9,'input10':input10}
            Y_train = Y_train[0:length_adapted_batch_size]

            number_workers_validation = 8
            number_files_per_worker = len(subject_files_val)//number_workers_validation
            X_validate, Y_validate = multi_processing(subject_files_val,number_files_per_worker,number_workers_validation)
            length_validate = Y_validate.shape[0]
            length_adapted_batch_size = closestNumber(length_validate-batch_size,batch_size)
            input1 = np.reshape(X_validate['input1'][0:length_adapted_batch_size],(length_adapted_batch_size,20,21,1))
            input2 = np.reshape(X_validate['input2'][0:length_adapted_batch_size],(length_adapted_batch_size,20,21,1))
            input3 = np.reshape(X_validate['input3'][0:length_adapted_batch_size],(length_adapted_batch_size,20,21,1))
            input4 = np.reshape(X_validate['input4'][0:length_adapted_batch_size],(length_adapted_batch_size,20,21,1))
            input5 = np.reshape(X_validate['input5'][0:length_adapted_batch_size],(length_adapted_batch_size,20,21,1))
            input6 = np.reshape(X_validate['input6'][0:length_adapted_batch_size],(length_adapted_batch_size,20,21,1))
            input7 = np.reshape(X_validate['input7'][0:length_adapted_batch_size],(length_adapted_batch_size,20,21,1))
            input8 = np.reshape(X_validate['input8'][0:length_adapted_batch_size],(length_adapted_batch_size,20,21,1))
            input9 = np.reshape(X_validate['input9'][0:length_adapted_batch_size],(length_adapted_batch_size,20,21,1))
            input10 = np.reshape(X_validate['input10'][0:length_adapted_batch_size],(length_adapted_batch_size,20,21,1))
            X_validate = {'input1' : input1,'input2' : input2,'input3' : input3, 'input4': input4,'input5':input5,'input6' : input6,'input7' : input7,'input8' : input8, 'input9': input9,'input10':input10}
            Y_validate = Y_validate[0:length_adapted_batch_size]
            
            input1 = None
            input2 = None
            input3 = None
            input4 = None
            input5 = None
            input6 = None
            input7 = None
            input8 = None
            input9 = None
            input10 = None
            
            training_end_subject = time.time()
            
#            with tf.device('/device:GPU:0'):
            history = cascade_model.fit(X_train, Y_train, batch_size = batch_size, epochs = 1, 
                                    verbose = 2, validation_data=(X_validate, Y_validate), 
                                    callbacks=None)
            subj_train_timespan = training_end_subject - start_subject_time
            print("Timespan subject training is : {}".format(subj_train_timespan))
            history_dict = history.history
            accuracies_temp_train.append(history_dict['acc'][0])#its because its a list of 1 element
            losses_temp_train.append(history_dict['loss'][0])
            accuracies_temp_val.append(history_dict['val_acc'][0])
            losses_temp_val.append(history_dict['val_loss'][0])
            #Freeing memory
            X_train = None
            Y_train = None
            X_validate = None
            Y_validate = None
            gc.collect()
        
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


        if (epoch+1) % 2 == 0 :
            print("Testing on subjects")
            accuracies_temp = []
            #Creating dataset for testing
            for subject in list_subjects_test:
                print("Reading data from subject", subject)
                subject_files_test = []
                for item in test_files_dirs:
                        if subject in item:
                            subject_files_test.append(item)
                            
                number_workers_testing = 10
                number_files_per_worker = len(subject_files_test)//number_workers_testing
                print(number_files_per_worker)
                X_test, Y_test = multi_processing(subject_files_test,number_files_per_worker,number_workers_testing)

                print("\n\nEvaluation cross-subject: ")
                result = cascade_model.evaluate(X_test, Y_test, batch_size = batch_size)
                
                accuracies_temp.append(result[1])
                print("Recording the testing accuracy of '{}' in a file".format(subject))
                append_individual_test(experiment_number,epoch,subject,result[1])
            avg = sum(accuracies_temp)/len(accuracies_temp)
            print("Average testing accuracy : {0:.2f}".format(avg))
            print("Recording the average testing accuracy in a file")
            append_average_test(experiment_number,epoch,avg)

            X_test = None
            Y_test = None
                
    stop_time = time.time()
    time_span = stop_time - start_time
    print()
    print()
    print("training took {:.2f} seconds".format(time_span))
    model_save(experiment_number,cascade_model)
    on_train_end(experiment_number)
    save_training_time(experiment_number, time_span)
    write_comment(experiment_number,comment)

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-s', '--setup', type=int, 
    help="Please select a number between 0 and 4 to choose \
    the setup of the training")

args = parser.parse_args()


if (args.setup):
    print("You chose setup {}".format(args.setup))
    hybrid_training(args.setup)
else:
    print("No setup has been chosen, basic training will start ...")
    print("Training Setup : 0")
    setup = 0
    hybrid_training(setup)
    
    
    
       



