# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 03:01:00 2019

@author: Smail
"""

import numpy as np
number_channels = 248
test = np.arange(1,number_channels,dtype=np.int32)
test = np.reshape(test,(1,test.shape[0]))

#print(test.shape)
#print(test)
output = np.zeros((22,22),dtype=np.int32)
count = 1
j = int(output.shape[0]/2)
#for i in range(int(output.shape[0]/2)):
#    output[i,j:j+count] = test[i,
#  count += 2
#  j -= 1
#
##print(output)
#print(test[0,5:5+1])
output[0,10:10+1] = test[0,0:0+1]
output[1,9:9+3] = test[0,1:1+3]
output[2,8:8+5] = test[0,4:4+5]
output[3,7:7+7] = test[0,9:9+7]
output[4,6:6+9] = test[0,16:16+9]

print(output)


#j = int(output.shape[0]/2) - 1
#k = 1
#l = 0
#for i in range(int(output.shape[0]/2)):
#    output[i,j:j+k] = test[0,l:l+k]
#    j -= 1
#    l += k
#    k += 2

i = 10
j = 11
k = 23
l = 121
#output[10,11:11+23] = test[0,121:121+23]
#print(len(output[10,0:22]))
#print(len(test[0,121:121+22]))
output[10,0:0 + 22] = test[0,121:121+22]
#output[11,1:1 + 20] = test[0,121+22:121+22+20]
#output[12,2:2 + 18] = test[0,]

#print(len(output[11,1:1 + 20])) #= 
#print(output[11,1:1 + 20])
#print(len(test[0,121+22:121+22+20]))
#print(test[0,121+22:121+22+20])
#output[12,2:20] = test[0,118:118+18]








    
#print(output)





















