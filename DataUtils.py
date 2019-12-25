import sys
import h5py
import boto3
import shutil
import os.path as op
import numpy as np
import os
import hcp
import mne


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
      
def get_raw_data(subject, type_state, hcp_path):
  try: #type of state for this subject might not exist
    print("Reading the binary file and returning the raw matrix ...")
    raw = hcp.read_raw(subject=subject, hcp_path=hcp_path, run_index=0, data_type=type_state)
    raw.load_data()
    meg_picks = mne.pick_types(raw.info, meg=True, ref_meg=False)
    return raw[meg_picks[:]][0]
  except Exception as e:
    print("Problem in reading the file: The type of state '{}' might not be there for subject '{}'".format(type_state, subject))
    print("Exception error : ",e)
    return False

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

def compare_raw_h5(matrix_raw_data,h5_file_name):
  byte_size_raw, mean_raw, max_raw = get_info_matrix(matrix_raw_data)
  byte_size_h5, mean_h5, max_h5 = get_info_h5_file(h5_file_name)
  print("difference in means: ", mean_raw - mean_h5)
  print("difference in maxes: ", max_raw - max_h5)
  print("difference in byte sizes: ", byte_size_raw - byte_size_h5)
  
def get_h5_file(h5_file_name):
  hf = h5py.File(h5_file_name, 'r')
  temp = h5_file_name.split('.') # we assume here that the dataset name is the same as the file name, except the extension
  dataset_name = temp[0]
  matrix = hf.get(dataset_name)
  matrix = np.array(matrix)
  return matrix

## TEST THE STOP INDEX , Might not be Working !! ##
def create_h5_files(raw_matrix,subject,type_state):
  train_folder = "../data/train/"
  validate_folder = "../data/validate/"
  test_folder = "../data/test/"

  if(type_state == "rest"): #we divide the file by 12 parts
    number_columns_per_chunk = closestNumber(raw_matrix.shape[1] - 12,12) // 12
    for i in range(12):
      if i >= 0 and i < 7:
        hf = h5py.File(train_folder + type_state + '_' + subject + '_' + str(i+1)+'.h5', 'w')
      if i >= 7 and i < 10:
        hf = h5py.File(validate_folder + type_state + '_' + subject + '_' + str(i+1)+'.h5', 'w')
      else:
        hf = h5py.File(test_folder + type_state + '_' + subject + '_' + str(i+1)+'.h5', 'w')
      start_index_col = number_columns_per_chunk * i
      stop_index_col = start_index_col + number_columns_per_chunk - 1
      hf.create_dataset(type_state+'_'+subject, data=raw_matrix[ : , start_index_col : stop_index_col ],compression="gzip", compression_opts=4) # lossless compression :) , 4 is the best option
      hf.close()

  if(type_state == "task_working_memory"): #we divide the file by 24 parts
    number_columns_per_chunk = closestNumber(raw_matrix.shape[1] - 24,24) // 24
    for i in range(24):
      if i>= 0 and i<14:
        hf = h5py.File(train_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5', 'w')
      if i >= 14 and i<19:
        hf = h5py.File(validate_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5', 'w')
      else:
        hf = h5py.File(test_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5', 'w')
      start_index_col = number_columns_per_chunk * i
      stop_index_col = start_index_col + number_columns_per_chunk - 1
      hf.create_dataset(type_state+'_'+subject, data=raw_matrix[ : , start_index_col : stop_index_col ],compression="gzip", compression_opts=4) # lossless compression :) , 4 is the best option
      hf.close()

  if(type_state == "task_story_math"): #we divide the file by 20 parts
    number_columns_per_chunk = closestNumber(raw_matrix.shape[1] - 20,20) // 20
    for i in range(20):
      if i>= 0 and i < 12:
        hf = h5py.File(train_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5', 'w')
      if i <= 12 and i < 16:
        hf = h5py.File(validate_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5', 'w')
      else:
        hf = h5py.File(test_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5', 'w')
      start_index_col = number_columns_per_chunk * i
      stop_index_col = start_index_col + number_columns_per_chunk - 1
      hf.create_dataset(type_state+'_'+subject, data=raw_matrix[ : , start_index_col : stop_index_col ],compression="gzip", compression_opts=4) # lossless compression :) , 4 is the best option
      hf.close()

  if(type_state == "task_motor"): #we divide the file by 30 parts
    number_columns_per_chunk = closestNumber(raw_matrix.shape[1] - 30,30) // 30
    for i in range(30):
      if i >= 0 and i < 18:
        hf = h5py.File(train_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5', 'w')
      if i >= 18 and i < 24:
        hf = h5py.File(validate_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5', 'w')
      else:
        hf = h5py.File(test_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5', 'w')
      hf = h5py.File(type_state+'_'+subject+'_'+str(i+1)+'.h5', 'w')
      start_index_col = number_columns_per_chunk * i
      stop_index_col = start_index_col + number_columns_per_chunk - 1
      hf.create_dataset(type_state+'_'+subject, data=raw_matrix[ : , start_index_col : stop_index_col ],compression="gzip", compression_opts=4) # lossless compression :) , 4 is the best option
      hf.close()

  if(type_state == "test"):
    test_matrix = np.zeros((10,10))
    hf = h5py.File('test.h5', 'w')
    hf.create_dataset('test', data=test_matrix,compression="gzip", compression_opts=4)
    hf.close()
    
def download_subject(subject,personal_access_key_id,secret_access_key):
  s3 = boto3.client('s3', aws_access_key_id=personal_access_key_id, aws_secret_access_key=secret_access_key)

  folders = ["3-Restin","4-Restin","5-Restin","6-Wrkmem","7-Wrkmem","8-StoryM","9-StoryM","10-Motort","11-Motort"]
  filename_test = ["e,rfhp1.0Hz,COH"]
  #filenames = ["c,rfDC", "config", "e,rfhp1.0Hz,COH", "e,rfhp1.0Hz,COH"]

  print("Creating the directories for the subject '{}'".format(subject))
  if op.exists(os.getcwd()+"//"+subject) == False:
    for folder in folders:
        os.makedirs(subject+"/unprocessed/MEG/"+folder+"/4D/")
  print("done !")
  print("Will start downloading the following files for all folders:")
  # print(filenames)
  print(filename_test)
  for filename in filename_test:
    for folder in folders:
      if filename == "c,rfDC":
        print("downloading c,rfDC file for folder {} ...".format(folder))
      if(op.exists(os.getcwd()+"//"+subject+'/unprocessed/MEG/'+folder+'/4D/'+filename)):
        print("File already exists, moving on ...")
        pass
      try:
        s3.download_file('hcp-openaccess', 'HCP_1200/'+subject+'/unprocessed/MEG/'+folder+'/4D/'+filename, subject+'/unprocessed/MEG/'+folder+'/4D/'+filename)
        if filename == "c,rfDC":
          print("done downloading c,rfDC for folder {} !".format(folder))
      except Exception as e:
        print("the folder '{}' for subject '{}' does not exist, moving to next folder ...".format(folder,subject))
        print("Exception error message: "+str(e))
        pass
    

    
def download_batch_subjects(list_subjects, personal_access_key_id, secret_access_key, hcp_path): # hcp_path should be os.getcwd()
  state_types = ["rest", "task_working_memory", "task_story_math", "task_motor"]
  for subject in list_subjects:
    download_subject(subject,personal_access_key_id,secret_access_key)
    for state in state_types:
      matrix_raw = get_raw_data(subject, state, hcp_path)
      if type(matrix_raw) != type(False): # if the reading was done successfully
        create_h5_files(matrix_raw,subject,state)
    print("done creating the compressed h5 files for subject '{}' !".format(subject))
    print("deleting the directory containing the binary files of subject '{}' ...")
    try:
      shutil.rmtree(subject+"/",ignore_errors=True)
      print("Done deleting the directory !")
      print("Moving on to the next subject ...")
    except Exception as e :
      print("Error while trying to delete the directory.")
      print("Exception message : " + str(e))
    
