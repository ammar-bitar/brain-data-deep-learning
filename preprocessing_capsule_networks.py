# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 03:01:00 2019

@author: Smail
"""

import numpy as np

def array_to_mesh(array_channels):
    assert array_channels.shape == (1,248),"the shape of the input array should be (1,248) because there are 248 MEG channels"
    output = np.zeros((20,22),dtype = np.float)
    output[0,9:9+3] = array_channels[0,0:0+3]
    j = int(output.shape[1]/2) - 2
    k = 3
    l = 3
    for i in range(1,int(output.shape[1]/2)):
        output[i,j:j+k] = array_channels[0,l:l+k]
        if( i < 10):
            j -= 1
            l += k
            k += 2
    k = k +1
    output[10,j:j + k] = array_channels[0,l:l+k]
    l += k
    row = 11
    for i in range(8):
        output[row,i:i+k] = array_channels[0,l:l+k]
        row += 1   
        l += k
        k -= 2
    i += 2
    k-=2
    output[row,i:i+k] = array_channels[0,l:l+k]
    return output
    

#####
## testing function
    
number_channels = 248
test = np.arange(0,number_channels,dtype=np.int32)
test[0] = -1 # for visibility, making the first element as -1 because of the 0s
test_input = np.reshape(test,(1,test.shape[0]))
output = array_to_mesh(test_input)
print(output)
    
    
    




















