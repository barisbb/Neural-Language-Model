import numpy as np
import math


def sigmoid(value):
    return 1 / (1 + np.exp(-value))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_loss(output,train_labels):
    return -math.log(output[0,train_labels])

def one_hot(train_index):
    vector=np.zeros((1,250))
    vector[0,train_index]=1
    return vector

def initialization():
    weights1=np.random.randn(250, 16) * 0.01

    weights2_1=np.random.randn(16, 128) * 0.01
    biases2_1=np.zeros(shape=(1, 128))
    weights2_2=np.random.randn(16, 128) * 0.01
    biases2_2=np.zeros(shape=(1, 128))
    weights2_3=np.random.randn(16, 128) * 0.01
    biases2_3=np.zeros(shape=(1, 128))

    weights3=np.random.randn(128, 250) * 0.01
    biases3=np.zeros(shape=(1, 250))
    return weights1, weights2_1,biases2_1,weights2_2,biases2_2,weights2_3,biases2_3,weights3,biases3

def forward_propagation(train, weights1,weights2_1,weights2_2,weights2_3,weights3,biases2_1,biases2_2,biases2_3,biases3):
    
        e1=np.dot(one_hot(train[0]),weights1)
        e2=np.dot(one_hot(train[1]),weights1)
        e3=np.dot(one_hot(train[2]),weights1)

        h1=np.dot(e1,weights2_1)+biases2_1
        h2=np.dot(e2,weights2_2)+biases2_2
        h3=np.dot(e3,weights2_3)+biases2_3

        l1=h1+h2+h3
        l1=sigmoid(l1)
        
        output=np.dot(l1,weights3)+biases3
        output_softmax=softmax(output)
        return output_softmax

def forward_backward_propagation(train,train_labels,learning_rate,weights1,weights2_1,weights2_2,weights2_3,weights3,biases2_1,biases2_2,biases2_3,biases3):
    
        e1=np.dot(one_hot(train[0]),weights1)
        e2=np.dot(one_hot(train[1]),weights1)
        e3=np.dot(one_hot(train[2]),weights1)

        h1=np.dot(e1,weights2_1)+biases2_1
        h2=np.dot(e2,weights2_2)+biases2_2
        h3=np.dot(e3,weights2_3)+biases2_3

        l1=h1+h2+h3
        l1=sigmoid(l1)
        
        output=np.dot(l1,weights3)+biases3
        output_softmax=softmax(output)
           
        weights3_before=np.copy(weights3) 
        weights2_1_before=np.copy(weights2_1)     
        weights2_2_before=np.copy(weights2_2)
        weights2_3_before=np.copy(weights2_3)
       
       
        
        weights3=weights3-learning_rate*np.dot(l1.T,(output_softmax-one_hot(train_labels)))
        biases3=biases3-learning_rate*(output_softmax-one_hot(train_labels))
        
        gradient=np.dot((output_softmax-one_hot(train_labels)),weights3_before.T)*l1*(1-l1)
        
        weights2_1=weights2_1-learning_rate*np.dot(e1.T,gradient)
        weights2_2=weights2_2-learning_rate*np.dot(e2.T,gradient)
        weights2_3=weights2_3-learning_rate*np.dot(e3.T,gradient)
        
        biases2_1=biases2_1-learning_rate*gradient
        biases2_2=biases2_2-learning_rate*gradient
        biases2_3=biases2_3-learning_rate*gradient
        
        weights1=weights1-learning_rate*(np.dot(one_hot(train[0]).T,np.dot(gradient,weights2_1_before.T))+np.dot(one_hot(train[1]).T,np.dot(gradient,weights2_2_before.T))+np.dot(one_hot(train[2]).T,np.dot(gradient,weights2_3_before.T)))
        
        return weights1,weights2_1,weights2_2,weights2_3,weights3,biases2_1,biases2_2,biases2_3,biases3
        