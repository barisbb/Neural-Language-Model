#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import math
import Network as nt

def accuracy(array,array_labels, weights1,weights2_1,weights2_2,weights2_3,weights3,biases2_1,biases2_2,biases2_3,biases3):
    count=0
    for i in range(array.shape[0]):
        output_softmax=nt.forward_propagation(array[i,:], weights1,weights2_1,weights2_2,weights2_3,weights3,biases2_1,biases2_2,biases2_3,biases3)
        if(np.argmax(output_softmax)==array_labels[i]):
            count=count+1
    return count/array.shape[0]

test=np.load("data/data/test_inputs.npy")
test_labels=np.load("data/data/test_targets.npy")

model=np.load("model.npz")
weights1=model['model_1']
weights2_1=model['model_2']
weights2_2=model['model_3']
weights2_3=model['model_4']
weights3=model['model_5']
biases2_1=model['model_6']
biases2_2=model['model_7']
biases2_3=model['model_8']
biases3=model['model_9']

test_accuracy=accuracy(test,test_labels, weights1,weights2_1,weights2_2,weights2_3,weights3,biases2_1,biases2_2,biases2_3,biases3)

print("Test Accuracy:"+str(test_accuracy))







