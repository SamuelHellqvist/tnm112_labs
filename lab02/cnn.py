import numpy as np
from scipy import signal
import skimage
import data_generator

# Different activations functions
def activation(x, activation):
    if activation == 'linear':
        return x
    elif activation == 'relu':
        return np.maximum(0, x)
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif activation == 'softmax':
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    else:
        raise ValueError("Activation function is not valid", activation)

# 2D convolutional layer
def conv2d_layer(h,     # activations from previous layer, shape = [channels prev. layer, height, width]
                 W,     # conv. kernels, shape = [channels this layer, channels prev. layer, kernel height, kernel width]
                 b,     # bias vector
                 act    # activation function
):

    h = np.asarray(h)
    W = np.asarray(W)
    b = np.asarray(b)

    # h måste vara 3 eftersom vi har 3 kanaler, rgb

    # w måster vara 4. den har alla filter. 4 dimensioner.
    # CO = number of output channels
    # CI = number of input channels
    # KH = kernel height
    # KW = kernel width

    if h.ndim != 3 or W.ndim != 4:
        raise ValueError("Expected h as CHW and W as OIHW arrays")

    CO, CI, _, _ = W.shape
    if h.shape[0] != CI or b.shape != (CO,):
        raise ValueError("Input channels or bias dimensions do not match weights")

    ho = np.zeros((CO, h.shape[1], h.shape[2]),
                  dtype=np.result_type(h.dtype, W.dtype, b.dtype))
    for output_channel in range(CO):
        for input_channel in range(CI):
            kernel = np.flip(W[output_channel, input_channel], axis=(0, 1))
            ho[output_channel] += signal.convolve2d(
                h[input_channel], kernel, mode='same'
            )

    ho += b[:, np.newaxis, np.newaxis]
    return activation(ho, act)


# 2D max pooling layer
def pool2d_layer(h):  # activations from conv layer, shape = [channels, height, width]
    # 1. Specify the height and width of the output
    # varför delar vi på 2? Varje pooling matris ska bli "halva"
    sy, sx = h.shape[1] // 2, h.shape[2] // 2

    # 2. Specify array to store output
    ho = np.zeros((h.shape[0], sy, sx), dtype=h.dtype)

    # 3. Perform pooling for each channel.
    # pooling is done to reduce computations and memory use
    for c in range(h.shape[0]):
        ho[c] = skimage.measure.block_reduce(h[c], (2, 2), np.max)

    # now when pooling is done, we return the output
    return ho


# Flattening layer
def flatten_layer(h):  # activations from conv/pool layer, shape = [channels, height, width]
    # we flatten the array to a vector output, this makes operations easier in the dense layer
    # vi tar all data vi har och gör det till en array med en dimension.
    # nu kan vi använda det i fully connected layer.
    h = h.flatten()
    return h


# Dense (fully-connected) layer
def dense_layer(h,   # Activations from previous layer
                W,   # Weight matrix
                b,   # Bias vector
                act  # Activation function
):
    # Ensure that h is a column vector [Kx1]
    h = h.reshape(-1, 1)

    # Ensure that b is a column vector
    b = b.reshape(-1, 1)

    # Matrix multiplication and bias
    z = np.dot(W, h) + b

    # Apply activation function
    ho = activation(z, act)

    return ho.reshape(-1)


#---------------------------------
# Our own implementation of a CNN
#---------------------------------
class CNN:
    def __init__(
        self,
        dataset,         # DataGenerator
        verbose=True     # For printing info messages
    ):
        self.verbose = verbose
        self.dataset = dataset

    # Set up the CNN from provided weights
    def setup_model(
        self,
        W,                   # List of weight matrices
        b,                   # List of bias vectors
        lname,               # List of layer names
        activation='relu'    # Activation function of layers
    ):
        self.activation = activation
        self.lname = lname

        self.W = W
        self.b = b

        # Total number of weights in the model
        # (convolutional kernels, weight matrices, and bias vectors)
        W_total = sum(w.size for w in W if w is not None)
        b_total = sum(bias.size for bias in b if bias is not None)

        self.N = W_total + b_total
        print('Number of model weights: ', self.N)

    # Feedforward through the CNN of one single image
    def feedforward_sample(self, h):

        # Loop over all the model layers
        for l in range(len(self.lname)):
            act = self.activation

            if self.lname[l] == 'conv':
                h = conv2d_layer(h, self.W[l], self.b[l], act)
            elif self.lname[l] == 'pool':
                h = pool2d_layer(h)
            elif self.lname[l] == 'flatten':
                h = flatten_layer(h)
            elif self.lname[l] == 'dense':
                if l == (len(self.lname) - 1):
                    act = 'softmax'
                h = dense_layer(h, self.W[l], self.b[l], act)
        return h

    # Feedforward through the CNN of a dataset
    def feedforward(self, x):
        # Output array
        y = np.zeros((x.shape[0], self.dataset.K))

        # Go through each image
        for k in range(x.shape[0]):
            if self.verbose and np.mod(k, 1000) == 0:
                print('sample %d of %d' % (k, x.shape[0]))

            # Apply layers to image
            y[k, :] = self.feedforward_sample(x[k])

        return y

    # Cross-entropy loss for integer class labels.
    # y_true: shape (N,) with integer labels 0..K-1
    # y_pred: shape (N, K) with softmax probabilities
    def cross_entropy(self, y_true, y_pred):
        epsilon = 1e-12
        y_true = np.asarray(y_true).astype(int)

        # Probability the model assigned to the correct class for each sample
        p_correct = y_pred[np.arange(len(y_true)), y_true]
        return -np.mean(np.log(p_correct + epsilon))

    # Measure performance of model
    def evaluate(self):
        print('Model performance:')

        # Training predictions
        y_pred = self.feedforward(self.dataset.x_train)

        # Training loss
        train_loss = self.cross_entropy(self.dataset.y_train, y_pred)

        # Training accuracy (labels are integers, so compare directly)
        train_acc = np.mean(np.argmax(y_pred, axis=1) == self.dataset.y_train)

        print("\tTrain loss:     %0.4f" % train_loss)
        print("\tTrain accuracy: %0.2f" % train_acc)

        # Test predictions
        y_pred_test = self.feedforward(self.dataset.x_test)

        # Test loss
        test_loss = self.cross_entropy(self.dataset.y_test, y_pred_test)

        # Test accuracy
        test_acc = np.mean(np.argmax(y_pred_test, axis=1) == self.dataset.y_test)

        print("\tTest loss:      %0.4f" % test_loss)
        print("\tTest accuracy:  %0.2f" % test_acc)