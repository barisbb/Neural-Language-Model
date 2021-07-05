#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import math
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import Network as nt

vocab=np.load("data/data/vocab.npy")

def one_hot(train_index):
    vector=np.zeros((1,250))
    vector[0,train_index]=1
    return vector

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


vocab_array=np.zeros((250,16))
for i in range(len(vocab)):
      vocab_array[i,:]=np.dot(one_hot(i),weights1)


# In[39]:


tsne_results = TSNE(n_components=2).fit_transform(vocab_array)


# In[40]:




fig, ax = plt.subplots()
ax.scatter(tsne_results[:,0], tsne_results[:,1],s=0.0001)

for i, txt in enumerate(vocab):
    ax.annotate(vocab[i], (tsne_results[i,0], tsne_results[i,1]))
    
ax.figure.savefig("scatter.png")


# In[37]:


city=np.where(vocab == "city")[0][0]
of=np.where(vocab == "of")[0][0]
new=np.where(vocab == "new")[0][0]
life=np.where(vocab == "life")[0][0]
in1=np.where(vocab == "in")[0][0]
the=np.where(vocab == "the")[0][0]
he=np.where(vocab == "he")[0][0]
is1=np.where(vocab == "is")[0][0]

word1=np.argmax( nt.forward_propagation([city,of,new], weights1,weights2_1,weights2_2,weights2_3,weights3,biases2_1,biases2_2,biases2_3,biases3) )
print("city of new "+vocab[word1])
word2=np.argmax( nt.forward_propagation([life,in1,the], weights1,weights2_1,weights2_2,weights2_3,weights3,biases2_1,biases2_2,biases2_3,biases3) )
print("life in the "+vocab[word2])
word3=np.argmax( nt.forward_propagation([he,is1,the], weights1,weights2_1,weights2_2,weights2_3,weights3,biases2_1,biases2_2,biases2_3,biases3) )
print("he is the "+vocab[word3])

