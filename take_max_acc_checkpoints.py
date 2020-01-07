import numpy as np
import os
import glob


def get_checkpoint_max_accuracy(directory):
    files = glob.glob(directory+"/*.hdf5")
    print("Found {} files in the directory {}".format(len(files),directory))
    accuracies = []
    for file in files:
        temp1 = file.split('_')
        last_part = temp1[-1]
        parts = last_part.split('.')
        first2 = parts[:2]
        accuracy_string = ".".join(item for item in first2)
        accuracies.append(float(accuracy_string))
        
    index_max = np.argmax(accuracies)
    file_max_accuracy = files[index_max]
    print("The file with the max accuracy is file {}".format(file_max_accuracy))
    return file_max_accuracy

# get_checkpoint_max_accuracy(os.getcwd())
    

