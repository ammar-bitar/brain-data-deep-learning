from tensorflow.keras.utils import Sequence
import h5py
from sklearn.utils import shuffle
import numpy as np
import gc
import data_utils as utils
import tensorflow as tf

class Generator(Sequence):
    def __init__(self, files_paths, ram_to_use, data_type,using_gpu): #ram_to_use in GB
        self.files_paths = files_paths
        # self.batch_size = batch_size
        self.index_batch = 0
        self.index_batch_files = 0
        self.ram_to_use = ram_to_use #not used anymore because unknown memory management of TF2 with/without GPU/CPU
        self.data_type = data_type # will be train or validate or test
        self.using_gpu = using_gpu
        self.number_files = self.get_number_files_to_load()
        
        

    def get_number_files_to_load(self):
        if self.using_gpu:
            number_files = 2            
        else:
            number_files = 6

        return number_files

    def get_dataset_name(self,file_name_with_dir):
        filename_without_dir = file_name_with_dir.split('/')[-1]
        temp = filename_without_dir.split('_')[:-1]
        dataset_name = "_".join(temp)
        return dataset_name


    def load_data(self):
        rest_matrix = np.zeros((248,1))
        math_matrix = np.zeros((248,1))
        memory_matrix = np.zeros((248,1))
        motor_matrix = np.zeros((248,1))
        number_classes = 4

        files_to_load = self.files_paths[self.index_batch_files * self.number_files: (self.index_batch_files + 1) * self.number_files]

        for i in range(len(files_to_load)):
            if "rest" in files_to_load[i]:
                with h5py.File(files_to_load[i],'r') as f:
                    dataset_name = self.get_dataset_name(files_to_load[i])
                    matrix = f.get(dataset_name)
                    matrix = np.array(matrix)
                assert matrix.shape[0] == 248 , "This rest data does not have 248 channels, but {} instead".format(matrix.shape[0])
                rest_matrix = np.column_stack((rest_matrix, matrix))

            if "math" in files_to_load[i]:
                with h5py.File(files_to_load[i],'r') as f:
                    dataset_name = self.get_dataset_name(files_to_load[i])
                    matrix = f.get(dataset_name)
                    matrix = np.array(matrix)
                assert matrix.shape[0] == 248 , "This math data does not have 248 channels, but {} instead".format(matrix.shape[0])
                math_matrix = np.column_stack((math_matrix, matrix))
                
            if "memory" in files_to_load[i]:
                with h5py.File(files_to_load[i],'r') as f:
                    dataset_name = self.get_dataset_name(files_to_load[i])
                    matrix = f.get(dataset_name)
                    matrix = np.array(matrix)
                assert matrix.shape[0] == 248 , "This memory data does not have 248 channels, but {} instead".format(matrix.shape[0])
                memory_matrix = np.column_stack((memory_matrix, matrix))
                
            if "motor" in files_to_load[i]:
                with h5py.File(files_to_load[i],'r') as f:
                    dataset_name = self.get_dataset_name(files_to_load[i])
                    matrix = f.get(dataset_name)
                    matrix = np.array(matrix)
                assert matrix.shape[0] == 248 , "This motor data does not have 248 channels, but {} instead".format(matrix.shape[0])
                motor_matrix = np.column_stack((motor_matrix, matrix))


        x_rest,y_rest = self.preprocess_data_type(rest_matrix)
        y_rest = y_rest*0
        input1_rest,input2_rest,input3_rest,input4_rest,input5_rest = x_rest[0],x_rest[1],x_rest[2],x_rest[3],x_rest[4]
        
          
        x_math,y_math = self.preprocess_data_type(math_matrix)            
        input1_math,input2_math,input3_math,input4_math,input5_math = x_math[0],x_math[1],x_math[2],x_math[3],x_math[4]            

        x_mem,y_mem = self.preprocess_data_type(memory_matrix)
        y_mem = y_mem * 2
        input1_mem,input2_mem,input3_mem,input4_mem,input5_mem = x_mem[0],x_mem[1],x_mem[2],x_mem[3],x_mem[4] 

        x_motor,y_motor = self.preprocess_data_type(motor_matrix)
        y_motor = y_motor * 3
        input1_motor,input2_motor,input3_motor,input4_motor,input5_motor = x_motor[0],x_motor[1],x_motor[2],x_motor[3],x_motor[4] 

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
                
            if dict2[i].shape[0]>0:
                input2=np.concatenate((input2,dict2[i]))
                
            if dict3[i].shape[0]>0:
                input3=np.concatenate((input3,dict3[i]))
                
            if dict4[i].shape[0]>0:
                input4=np.concatenate((input4,dict4[i]))
                
            if dict5[i].shape[0]>0:
                input4=np.concatenate((input4,dict5[i]))
                
                
        input1 = np.delete(input1,0,0)
        input2 = np.delete(input1,0,0)
        input3 = np.delete(input1,0,0)
        input4 = np.delete(input1,0,0)
        input5 = np.delete(input1,0,0)
        


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

        #Shuffling the data
        input1,input2,input3,input4,input5,y = shuffle(input1,input2,input3,input4,input5, y, random_state=42)

        data_dict = {'input1' : input1,'input2' : input2,'input3' : input3, 'input4': input4,'input5':input5}
        
        del input1
        del input2
        del input3
        del input4
        del input5

        gc.collect()
        
        y = tf.keras.utils.to_categorical(y,number_classes)
        return data_dict,y
    
    def preprocess_data_type(self,matrix):
        matrix = self.normalize_matrix(matrix)
        length = utils.closestNumber(int(matrix.shape[1]) - 10,10)
        meshes = np.zeros((length,20,22))
        for i in range(length):
            array_time_step = np.reshape(matrix[:,i],(1,248))
            meshes[i] = utils.array_to_mesh(array_time_step)

        del matrix
        gc.collect()
        input1,input2,input3,input4,input5 = self.get_input_lists(meshes, self.get_lists_indexes(length))
        del meshes
        gc.collect()
        number_y_labels = int(length/5)
        y_rest = np.ones((number_y_labels,1),dtype=np.int8)
        return (input1,input2,input3,input4,input5), y_rest
        


    def normalize_matrix(self,matrix):
        max,min = matrix.max(),matrix.min()
        return (matrix-min)/(max-min)

    def get_lists_indexes(self,matrix_length):
        indexes_1 = np.arange(start=0,stop = matrix_length-4,step=5)
        indexes_2 = np.arange(start=1,stop = matrix_length-3,step=5)
        indexes_3 = np.arange(start=2,stop = matrix_length-2,step=5)
        indexes_4 = np.arange(start=3,stop = matrix_length-1,step=5)
        indexes_5 = np.arange(start=4,stop = matrix_length-0,step=5)
        return (indexes_1,indexes_2,indexes_3,indexes_4,indexes_5)
    
    def get_input_lists(self,matrix,indexes):
        inputs = []
        for i in range(5):
            inputs.append(np.take(matrix,indexes[i],axis=0))
        return inputs[0],inputs[1],inputs[2],inputs[3],inputs[4]

    def on_epoch_end(self):
        self.index_batch_files = 0

    def __len__(self): # how many batches
        return len(self.files_paths)//self.number_files
        # return math.ceil(len(self.files_paths) / self.number_files)

    def __getitem__(self, index): # returns a batch
        gc.collect()
        batch_input, batch_output = self.load_data()
        # batch_input, batch_output = self.load_synthetic_data()
        self.index_batch_files += 1
        return batch_input,batch_output
    
    def to_one_hot_binary(self, array):
        output = np.zeros((array.shape[0],2),dtype=np.int)
        for i in range(len(array)):
            if array[i] == 0:
                output[i] = np.array([1,0])
            else:
                output[i] = np.array([0,1])
        return output
    



#load X files from train and 0.5X files from validate
#from training data, prepare all 5 input lists and output list
#do the same for validate data
#approximately 1 GB in RAM = 6 train + 3 validate
#approximately 2 GB in RAM = 13 train + 6 validate
#3 GB in RAM = 20 train + 10 validate
#5GB in RAM = 33 train + 16 validate
#batch size = no longer known ( and needed )


