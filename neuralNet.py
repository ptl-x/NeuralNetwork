import numpy as np

class NeuralNetwork(object):

  #constructor sets up hidden layers and network nodes
  def __init__(self, inputs, hidden_layers, outputs):
    
    #seed the neural net for consistency in testing
    np.random.seed(1)

    #create space for the layers
    self.layers = [None] * (len(hidden_layers) + 1)

    #generate the layers
    for i in range(len(self.layers)):
      if(i == 0):
         self.layers[0] = 2 * np.random.random((inputs, hidden_layers[i])) - 1
      elif(i == len(hidden_layers)):
         self.layers[i] = 2 * np.random.random((hidden_layers[i - 1], outputs)) - 1
      else:
         self.layers[i] = 2 * np.random.random((hidden_layers[i - 1], hidden_layers[i])) - 1

    #show the layers
    for layer in self.layers:
      print(layer, "\n")


  #node activation function
  def sigmoid(self, x):
      return 1 / (1 + np.exp(-x))


  #derivative of activation function used for back-propogation
  def sigmoid_derivative(self, x):
    return x * (1 - x)


  #forward-propogation method
  def think(self, x):
    o2 = self.sigmoid(np.dot(x, self.layers[0]))
    for i in range(len(self.layers) - 1):
      o = self.sigmoid(np.dot(o2, self.layers[i + 1]))
      o2 = o
    return o


  #training method using back-propogation
  def train(self, xi, xf, iterations):
    
    for iteration in range(iterations):
      
      #create an array to hold outputs and errors for each layer
      outs = [None] * (len(self.layers))
      for i in range(len(outs)):
        outs[i] = [None] * 3

      #forward-propogation of input data
      for i in range(len(self.layers)):
        if(i == 0):
          outs[0][0] = self.sigmoid(np.dot(xi, self.layers[i]))
        else:
          outs[i][0] = self.sigmoid(np.dot(outs[i - 1][0], self.layers[i]))

      #calculate and store the error and error delta for each layer
      for i in range(len(self.layers)):
        if(i == 0):
          outs[len(outs) - 1][1] = xf - outs[len(outs) - 1][0]
          outs[len(outs) - 1][2] = outs[len(outs) - 1][1] * self.sigmoid_derivative(outs[len(outs) - 1][0])
        else:
          outs[len(outs) - 1 - i][1] = outs[len(outs) - i][2].dot(self.layers[len(self.layers) - i].T)
          outs[len(outs) - 1 - i][2] = outs[len(outs) - 1 - i][1] * self.sigmoid_derivative(outs[len(outs) - 1 - i][0])

      #print out square of errors
      if(iteration % 1000 == 0):
        square_errors = 0
        for array in outs:
          square_errors += np.sum(array[1] * array[1])
        print(square_errors)

      #adjust each layer acording to the error delta
      for i in range(len(outs)):
        if(i == 0):
          self.layers[0] += xi.T.dot(outs[0][2])
        else:
          self.layers[i] += outs[i - 1][0].T.dot(outs[i][2])
