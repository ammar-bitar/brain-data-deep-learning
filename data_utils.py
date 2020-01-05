import sys
import h5py
import boto3
import shutil
import os.path as op
import numpy as np
import os
import mne
import reading_raw

#Given the number "n", it finds the closest that is divisible by "m"
#Used when splitting the matrices
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


def show_octet_size_file(filename):
  with open(filename,'rb') as f:
      input_stream = f.read()
      print(len(input_stream))
      
#Reads the binary file given the subject and the type of state
#If the reading is successful, returns the matrix of size (248,number_time_steps)
#If the reading is NOT successful, prints the problem and returns the boolean False
def get_raw_data(subject, type_state, hcp_path):
  try: #type of state for this subject might not exist
    print("Reading the binary file and returning the raw matrix ...")
    raw = reading_raw.read_raw(subject=subject, hcp_path=hcp_path, run_index=0, data_type=type_state)
    raw.load_data()
    meg_picks = mne.pick_types(raw.info, meg=True, ref_meg=False)
    raw_matrix = raw[meg_picks[:]][0]
    del raw
    return raw_matrix
  except Exception as e:
    print("Problem in reading the file: The type of state '{}' might not be there for subject '{}'".format(type_state, subject))
    print("Exception error : ",e)
    return False

#Generic function that returns information about any matrix
#Used to make sure that compression is lossless
def get_info_matrix(matrix):
  print("shape of matrix: ",matrix.shape)
  print("data type of matrix: ",matrix.dtype)
  print("byte size of matrix data type: ", sys.getsizeof(matrix.dtype))
  byte_size = sys.getsizeof(matrix)
  print("byte size of matrix : ",byte_size)
  mean = np.mean(matrix)
  maximum = np.max(matrix)
  print("mean of matrix: ",mean)
  print("max of matrix: ", maximum)
  return byte_size,mean,maximum
  
#Generic function that returns information about the matrix of any h5 file
def get_info_h5_file(h5_file_name):
  hf = h5py.File(h5_file_name, 'r')
  temp = h5_file_name.split('.') # we assume here that the dataset name is the same as the file name, except the extension
  dataset_name = temp[0]
  matrix = hf.get(dataset_name)
  matrix = np.array(matrix)
  print("Shape of uncompressed h5 matrix: ",matrix.shape)
  byte_size_matrix = sys.getsizeof(matrix)
  print("byte size of uncompressed h5 matrix: ", byte_size_matrix)
  mean_reading = np.mean(matrix)
  max_reading = np.max(matrix)
  print("mean of matrix: ",mean_reading)
  print("max of matrix: ", max_reading)
  hf.close()
  return byte_size_matrix, mean_reading, max_reading

#Function to compare the output matrix of a h5 file with the original binary matrix output of read_raw() function
#Used to make sure that compression is lossless
def compare_raw_h5(matrix_raw_data,h5_file_name):
  byte_size_raw, mean_raw, max_raw = get_info_matrix(matrix_raw_data)
  byte_size_h5, mean_h5, max_h5 = get_info_h5_file(h5_file_name)
  print("difference in means: ", mean_raw - mean_h5)
  print("difference in maxes: ", max_raw - max_h5)
  print("difference in byte sizes: ", byte_size_raw - byte_size_h5)
  
  
#Reads a h5 file and returns the matrix inside of it as a numpy matrix
#We assume there is only 1 dataset per h5 file and the dataset name has the same name of the file name
#See https://www.christopherlovell.co.uk/blog/2016/04/27/h5py-intro.html for more info on how h5 files work
def get_h5_file(h5_file_name,verbose=False):
  if(verbose):
      print("Uncompressing and Reading the file {}".format(h5_file_name))
  hf = h5py.File(h5_file_name, 'r')
  temp = h5_file_name.split('.') # we assume here that the dataset name is the same as the file name, except the extension
  dataset_name = temp[0]
  matrix = hf.get(dataset_name)
  matrix = np.array(matrix)
  return matrix

## TEST THE STOP INDEX , Might not be Working !! ## # Still need to be investigated, but it's a minor problem
#Function to create the h5 files given the raw matrix
#The splitting is done so that the memory size in RAM once *uncompressed* is about 100 MB
#Since the data files have different average size, the splitting will be different
#rest matrix is splitted into 12 parts
#task_working_memory matrix is splitted into 24 parts
#task_story_math is splitted into 20 parts
#task_motor is splitted into 30 parts
#
#Each part goes to 3 different folders : train, validate and test
#Splitting is : 60% train, 20% validate, 20% test
#rest matrix = 12 = 7 in train + 3 in validate + 2 in test
#task_working_memory matrix = 24 = 14 in train + 5 in validate + 5 in test
#task_story_math = 20 = 12 in train + 4 in validate + 4 in test
#task_motor = 30 = 18 in train + 6 in validate + 6 in test
#When not testing, change things related to filename_test variable
def create_h5_files(raw_matrix,subject,type_state):
    print()
    print("shape of raw matrix",raw_matrix.shape)
    print()
    train_folder = "Data/train/"
    validate_folder = "Data/validate/"
    test_folder = "Data/test/"

    if(type_state == "rest"): #we divide the file by 12 parts
        number_columns_per_chunk = closestNumber(raw_matrix.shape[1] - 12,12) // 12
        for i in range(12):
          if i >= 0 and i < 7:
              destination_file = train_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5'
          if i >= 7 and i < 10:
              destination_file = validate_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5'
          if i >= 10:
              destination_file = test_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5'
          start_index_col = number_columns_per_chunk * i
          stop_index_col = start_index_col + number_columns_per_chunk - 1
          with h5py.File(destination_file, "w") as hf:
              hf.create_dataset(type_state+'_'+subject, data=raw_matrix[ : , start_index_col : stop_index_col ],compression="gzip", compression_opts=4) # lossless compression :) , 4 is the best option
    
    if(type_state == "task_working_memory"): #we divide the file by 24 parts
        number_columns_per_chunk = closestNumber(raw_matrix.shape[1] - 24,24) // 24
        for i in range(5): # we choose only 5 chunks of this data to solve data imbalance
          if i>= 0 and i<3:
              destination_file = train_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5'
          if i == 3:
              destination_file = validate_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5'
          if i == 4:
              destination_file = test_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5'
          start_index_col = number_columns_per_chunk * i
          stop_index_col = start_index_col + number_columns_per_chunk - 1
          with h5py.File(destination_file, "w") as hf:
              hf.create_dataset(type_state+'_'+subject, data=raw_matrix[ : , start_index_col : stop_index_col ],compression="gzip", compression_opts=4) # lossless compression :) , 4 is the best option
    
    if(type_state == "task_story_math"): #we divide the file by 18 parts
        number_columns_per_chunk = closestNumber(raw_matrix.shape[1] - 18,18) // 18
        for i in range(5): # we choose only 5 chunks of this data to solve data imbalance
          if i >= 0 and i < 3:
              destination_file = train_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5'
          if i == 3:
              destination_file = validate_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5'
          if i == 4 :
              destination_file = test_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5'
          start_index_col = number_columns_per_chunk * i
          stop_index_col = start_index_col + number_columns_per_chunk - 1
          with h5py.File(destination_file, "w") as hf:
              hf.create_dataset(type_state+'_'+subject, data=raw_matrix[ : , start_index_col : stop_index_col ],compression="gzip", compression_opts=4) # lossless compression :) , 4 is the best option
    
    if(type_state == "task_motor"): #we divide the file by 30 parts
        number_columns_per_chunk = closestNumber(raw_matrix.shape[1] - 30,30) // 30
        for i in range(5):
          if i >= 0 and i < 3:
              destination_file = train_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5'
          if i == 3:
              destination_file = validate_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5'
          if i == 4:
              destination_file = test_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5'
          start_index_col = number_columns_per_chunk * i
          stop_index_col = start_index_col + number_columns_per_chunk - 1
          with h5py.File(destination_file, "w") as hf:
              hf.create_dataset(type_state+'_'+subject, data=raw_matrix[ : , start_index_col : stop_index_col ],compression="gzip", compression_opts=4) # lossless compression :) , 4 is the best option

    
    if(type_state == "test"):
        test_matrix = np.zeros((10,10))
        with h5py.File('test.h5', 'w') as hf:
            hf.create_dataset('test', data=test_matrix,compression="gzip", compression_opts=4)
            

##For each subject, it prints how many rest files and how many task files it has in the Amazon server            
def get_info_files_subjects(personal_access_key_id,secret_access_key, subjects):
    folders = ["3-Restin","4-Restin","5-Restin","6-Wrkmem","7-Wrkmem","8-StoryM","9-StoryM","10-Motort","11-Motort"]
    s3 = boto3.client('s3', aws_access_key_id=personal_access_key_id, aws_secret_access_key=secret_access_key)
    for subject in subjects:
        rest_count = 0
        task_count = 0
        for folder in folders:
            number_files = s3.list_objects_v2(Bucket="hcp-openaccess", Prefix='HCP_1200/'+subject+'/unprocessed/MEG/'+folder)['KeyCount']
            if "Restin" in folder:
                rest_count += number_files
            else:
                task_count += number_files
        print("for subject {}, rest_count = {}, and task_count = {}".format(subject, rest_count, task_count))
        
#Gets a list of subjects and returns a new list of subjects that might contain less subjects
#It iterates through all the subjects, and if a subject has 0 task or rest state files, it discards the subject
#Used to fix data imbalance and to prevent bugs during the training using the data generator
def get_filtered_subjects(personal_access_key_id,secret_access_key, subjects):
    print("starting to discard subjects to keep data balance ...")
    new_subject_list = []
    number_subjects = len(subjects)
    folders = ["3-Restin","4-Restin","5-Restin","6-Wrkmem","7-Wrkmem","8-StoryM","9-StoryM","10-Motort","11-Motort"]
    s3 = boto3.client('s3', aws_access_key_id=personal_access_key_id, aws_secret_access_key=secret_access_key)
    for subject in subjects:
        rest_count = 0
        task_count = 0
        for folder in folders:
            number_files = s3.list_objects_v2(Bucket="hcp-openaccess", Prefix='HCP_1200/'+subject+'/unprocessed/MEG/'+folder)['KeyCount']
            if "Restin" in folder:
                rest_count += number_files
            else:
                task_count += number_files
        if (rest_count != 0 and task_count !=0):
            new_subject_list.append(subject)
        else:
            if rest_count == 0:
                print("Discarding subject '{}' because it had 0 rest files".format(subject))
            else:
                print("Discarding subject '{}' because it had 0 task files".format(subject))

    new_list_len = len(new_subject_list)
    print("-"*7 + " Done filtering out subjects ! " + "-"*7)
    print("Original list had {} subjects and new list has {} subjects".format(number_subjects, new_list_len))
    return new_subject_list

###Downloads 1 subject and ignores the case where 1 of the folders/files is missing    
def download_subject(subject,personal_access_key_id,secret_access_key):
  s3 = boto3.client('s3', aws_access_key_id=personal_access_key_id, aws_secret_access_key=secret_access_key)

  folders = ["3-Restin","4-Restin","5-Restin","6-Wrkmem","7-Wrkmem","8-StoryM","9-StoryM","10-Motort","11-Motort"]
  #filename_test = ["e,rfhp1.0Hz,COH"]
  filenames = ["c,rfDC", "config", "e,rfhp1.0Hz,COH", "e,rfhp1.0Hz,COH1"]

  print("Creating the directories for the subject '{}'".format(subject))
  print()
  if op.exists(os.getcwd()+"//"+subject) == False:
    for folder in folders:
        os.makedirs(subject+"/unprocessed/MEG/"+folder+"/4D/")
  print("done !")
  print()
  print("Will start downloading the following files for all folders:")
  print(filenames)
  print()
  print()
  #print(filename_test)
  for filename in filenames:
    for folder in folders:
      if filename == "c,rfDC":
        print("downloading c,rfDC file for folder {} ...".format(folder))
        print()
      if(op.exists(os.getcwd()+"//"+subject+'/unprocessed/MEG/'+folder+'/4D/'+filename)):
        print("File already exists, moving on ...")
        print()
        pass
      try:
        s3.download_file('hcp-openaccess', 'HCP_1200/'+subject+'/unprocessed/MEG/'+folder+'/4D/'+filename, subject+'/unprocessed/MEG/'+folder+'/4D/'+filename)
        if filename == "c,rfDC":
          print("done downloading c,rfDC for folder {} !".format(folder))
          print()
      except Exception as e:
        print()
        print("the folder '{}' for subject '{}' does not exist in Amazon server, moving to next folder ...".format(folder,subject))
        print("Exception error message: "+str(e))
        pass
    

#Main function to be executed to download subjects
#list_subjects should be a list of strings containing the 6 digits subjects
#hcp_path should be the current working directory (os.getcwd())
def download_batch_subjects(list_subjects, personal_access_key_id, secret_access_key, hcp_path): # hcp_path should be os.getcwd()
  state_types = ["rest", "task_working_memory", "task_story_math", "task_motor"]
  for subject in list_subjects:
    download_subject(subject,personal_access_key_id,secret_access_key)
    for state in state_types:
      matrix_raw = get_raw_data(subject, state, hcp_path)
      if type(matrix_raw) != type(False): # if the reading was done successfully
        print()
        print("Creating the compressed h5 files ...")
        create_h5_files(matrix_raw,subject,state)
    print("done creating the compressed h5 files for subject '{}' !".format(subject))
    print()
    print("deleting the directory containing the binary files of subject '{}' ...".format(subject))
    print()
    try:
      shutil.rmtree(subject+"/",ignore_errors=True)#Removes the folder and all folders/files inside
      print("Done deleting the directory of the binary files!")
      print("Moving on to the next subject ...")
      print()
    except Exception as e :
      print()
      print("Error while trying to delete the directory.")
      print("Exception message : " + str(e))
    
