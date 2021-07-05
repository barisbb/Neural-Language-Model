#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import math
import random
import Network as nt


def accuracy(array,array_labels, weights1,weights2_1,weights2_2,weights2_3,weights3,biases2_1,biases2_2,biases2_3,biases3):
    count=0
    for i in range(array.shape[0]):
        output_softmax=nt.forward_propagation(array[i,:], weights1,weights2_1,weights2_2,weights2_3,weights3,biases2_1,biases2_2,biases2_3,biases3)
        if(np.argmax(output_softmax)==array_labels[i]):
            count=count+1
    return count/array.shape[0]
            
train=np.load("data/data/train_inputs.npy")
train_labels=np.load("data/data/train_targets.npy")

valid=np.load("data/data/valid_inputs.npy")
valid_labels=np.load("data/data/valid_targets.npy")


train_zip = list(zip(train, train_labels))
random.shuffle(train_zip)
train, train_labels = zip(*train_zip)
train=np.asarray(train)
train_labels=np.asarray(train_labels)


val_zip = list(zip(valid, valid_labels))
random.shuffle(val_zip)
valid, valid_labels = zip(*val_zip)
valid=np.asarray(valid)
valid_labels=np.asarray(valid_labels)

weights1, weights2_1,biases2_1,weights2_2,biases2_2,weights2_3,biases2_3,weights3,biases3=nt.initialization()

learning_rate=0.01
epoch=10
for epoch in range(epoch):
    count=0
    for j in range(train.shape[0]):    
           weights1,weights2_1,weights2_2,weights2_3,weights3,biases2_1,biases2_2,biases2_3,biases3=nt.forward_backward_propagation(train[j,:],train_labels[j],learning_rate,weights1,weights2_1,weights2_2,weights2_3,weights3,biases2_1,biases2_2,biases2_3,biases3)
    print("Epoch:"+ str(epoch+1)+"\n")
    print("    Training Accuracy:"+str(accuracy(train,train_labels, weights1,weights2_1,weights2_2,weights2_3,weights3,biases2_1,biases2_2,biases2_3,biases3)))
    print("    Validation Accuracy:"+str(accuracy(valid,valid_labels, weights1,weights2_1,weights2_2,weights2_3,weights3,biases2_1,biases2_2,biases2_3,biases3)))     
    
np.savez('model.npz', model_1=weights1,model_2=weights2_1,model_3=weights2_2,model_4=weights2_3,model_5=weights3,model_6=biases2_1,model_7=biases2_2,model_8=biases2_3,model_9=biases3)
    

