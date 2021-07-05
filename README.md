# Neural-Language-Model

This is an implementation of multi-layer perceptron without using deep learning frameworks.

It includes:

Network.py

main.py

eval.py

tsne.py

To train the model, you need to run main.py. There are some paths that you need to revise, which are the paths of training and validation set. Also, at the end of the code, you can revise the path that you want to save the model parameters. 

To evaluate your results on test set, you should revise the path of the test set and pretrained model parameters. After that, you can run eval.py.

In tsne.py, you should also revise the path of the pretrained model parameters and vocab file. Then, when you run the code, the figure of the t-sne result will be saved to the default path and the predictions of the following data points:city of new ..., life in the ..., he is the ... will be printed.

