from neuralNet import NeuralNetwork
imports numpy as np

#create a network with 3 inputs, 3 hidden layers (the first with 5 nodes
#the second with 5 nodes, and the third with 3 nodes), and a single output 
neuralNet = NeuralNetwork(3, [5, 5, 3], 1)

#create input data set
ni = np.array([[1, 0, 1],
               [1, 0, 0],
               [1, 1, 1],
               [0, 0, 0],
               [1, 1, 0],
               [0, 1, 0],
               [0, 0, 1],
               [0, 1, 1]])

#create output data set
no = np.array([[0],
               [1],
               [0],
               [0],
               [0],
               [1],
               [1],
               [0]])

#train network with data sets
neuralNet.train(ni, no, 100000)

#test an input set for accuracy
print(neuralNet.think(np.array([0, 1, 1])))
