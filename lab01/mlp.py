import numpy as np
import data_generator

# Different activations functions
def activation(x, activation):
    
    #TODO: specify the different activation functions
    # 'activation' could be: 'linear', 'relu', 'sigmoid', or 'softmax'
    if activation == 'linear':
        return x
    elif activation == 'relu':
        return np.maximum(0, x)
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif activation == 'softmax':
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    else:
        raise Exception("Activation function is not valid", activation) 

#-------------------------------
# Our own implementation of an MLP
#-------------------------------
class MLP:
    def __init__(
        self,
        dataset,         # DataGenerator
    ):
        self.dataset = dataset

    # Set up the MLP from provided weights and biases
    def setup_model(
        self,
        W,                   # List of weight matrices
        b,                   # List of bias vectors
        activation='linear'  # Activation function of layers
    ):
        self.activation = activation

        # TODO: specify the number of hidden layers based on the length of the provided lists
        self.hidden_layers = len(W) -1

        self.W = W
        self.b = b

        # TODO: specify the total number of weights in the model (both weight matrices and bias vectors)
        self.N = sum(w.size for w in W) + sum(b.size for b in b)

        print('Number of hidden layers: ', self.hidden_layers)
        print('Number of model weights: ', self.N)

    # Feed-forward through the MLP
    def feedforward(
        self,
        x      # Input data points
    ):
        # TODO: specify a matrix for storing output values
        y = np.zeros((x.shape[0], self.b[-1].shape[0]))

        # TODO: implement the feed-forward layer operations
        # 1. Specify a loop over all the datapoints
        for i in range(x.shape[0]):
            # 2. Specify the input layer (2x1 matrix)
            a = x[i].reshape(-1, 1)
            # 3. For each hidden layer, perform the MLP operations
            #    - multiply weight matrix and output from previous layer
            #    - add bias vector
            #    - apply activation function
            for j in range(self.hidden_layers):
                z = np.dot(self.W[j], a) + self.b[j]
                a = activation(z, self.activation)
            # 4. Specify the final layer, with 'softmax' activation
            z = np.dot(self.W[-1], a) + self.b[-1]
            y[i] = activation(z.T, 'softmax')[0]
        return y


    # Measure performance of model
    def evaluate(self):
        print('Model performance:')

        # TODO: formulate the training loss and accuracy of the MLP
        # Assume the mean squared error loss
        # Hint: For calculating accuracy, use np.argmax to get predicted class
        
        # the idea here is that we look at all the predicted values, and we 
        # compare them to the true labels. We count how many times the predicted 
        # class (the one with the highest probability) matches the true class, 
        # and then we divide by the total number of samples to get the accuracy.
        y_pred = self.feedforward(self.dataset.x_train)
        
        train_loss = np.mean(np.square(y_pred - self.dataset.y_train_oh))
        
        train_acc = np.mean(np.argmax(y_pred, axis=1) == self.dataset.y_train)
        
        print("\tTrain loss:     %0.4f"%train_loss)
        print("\tTrain accuracy: %0.2f"%train_acc)

        # TODO: formulate the test loss and accuracy of the MLP
        y_pred_test = self.feedforward(self.dataset.x_test)
        test_loss = np.mean(np.square(y_pred_test - self.dataset.y_test_oh))
        test_acc = np.mean(np.argmax(y_pred_test, axis=1) == self.dataset.y_test)
        print("\tTest loss:      %0.4f"%test_loss)
        print("\tTest accuracy:  %0.2f"%test_acc)
