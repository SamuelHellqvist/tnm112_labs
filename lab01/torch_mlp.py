import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


# -------------------------------
# PyTorch MLP
# -------------------------------
class TorchMLP(nn.Module):
    acts = {"linear": nn.Identity, "relu": nn.ReLU, "sigmoid": nn.Sigmoid}

    def __init__(
        self,
        K,  # Number of classes (outputs)
        hidden_layers=0,  # Number of hidden layers (total number of layers will be L=hidden_layers+1)
        layer_width=2,  # Number of neurons in each hidden layers
        activation="linear",  # Activation function
        init=None,  # Initialization method, a function that fills a weight tensor in place
        verbose=True,  # For printing info messages
    ):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.layer_width = layer_width

        if init is None:
            init = lambda w: nn.init.normal_(w, mean=0.0, std=0.05)

        layers = []
        n_in = 2  # 2-dimensional input
        for _ in range(hidden_layers):
            layers += [nn.Linear(n_in, layer_width), self.acts[activation]()]
            n_in = layer_width
        # Output layer, with softmax layer for classification
        layers += [nn.Linear(n_in, K), nn.Softmax(dim=1)]
        self.layers = nn.Sequential(*layers)

        # Weights from the specified initialization, biases start at zero
        for m in self.layers:
            if isinstance(m, nn.Linear):
                init(m.weight)
                nn.init.zeros_(m.bias)

        if verbose:
            print(self.layers)  # Model info
            print("Number of model weights: ", sum(p.numel() for p in self.parameters()))

    # Used when training: tensors in, tensors out
    def forward(self, x):
        return self.layers(x)

    # Feed-forward through the MLP: numpy in, numpy out (used by data.plot_classifier)
    def feedforward(self, x):  # Input data points
        self.eval()
        with torch.no_grad():
            p = self(torch.as_tensor(x, dtype=torch.float32))
        return p.numpy()

    # Extract weights and biases from model
    def get_weights(self):
        # Use only the fully-connected (linear) layers (activations have no weights)
        layers = [l for l in self.layers if isinstance(l, nn.Linear)]
        W = []
        b = []

        # Extract weight matrices and bias vectors
        for l in layers:
            W.append(l.weight.detach().numpy())  # already (out, in): no transpose
            b.append(l.bias.detach().numpy()[:, np.newaxis])  # column vector (out, 1)

        return W, b


# -------------------------------
# Data
# -------------------------------
# Wrap data (inputs x and one-hot labels y) in a DataLoader that provides it in batches
def make_loader(x, y, batch_size=32, shuffle=False):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# -------------------------------
# Training and testing
# -------------------------------
# One epoch of training. Returns the average loss and the accuracy over the epoch
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss, correct, N = 0.0, 0, 0
    for x, y in dataloader:
        p = model(x)
        loss = loss_fn(p, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x)
        correct += (p.argmax(1) == y.argmax(1)).sum().item()
        N += len(x)
    return total_loss / N, correct / N


# Loss and accuracy on a dataset, without training
def test(dataloader, model, loss_fn):
    model.eval()
    total_loss, correct, N = 0.0, 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            p = model(x)
            total_loss += loss_fn(p, y).item() * len(x)
            correct += (p.argmax(1) == y.argmax(1)).sum().item()
            N += len(x)
    return total_loss / N, correct / N


# Loss and accuracy on the training and test set
def evaluate(model, data, loss_fn):
    print("Model performance:")

    loss, acc = test(make_loader(data.x_train, data.y_train_oh, 512), model, loss_fn)
    print("\tTrain loss:     %0.4f" % loss)
    print("\tTrain accuracy: %0.2f" % (100 * acc))

    loss, acc = test(make_loader(data.x_test, data.y_test_oh, 512), model, loss_fn)
    print("\tTest loss:      %0.4f" % loss)
    print("\tTest accuracy:  %0.2f" % (100 * acc))


# Plot training log
def plot_training(log, save_path=None):
    plt.figure(figsize=(18, 4))

    # Plot loss on training and validation set
    plt.subplot(1, 2, 1)
    plt.plot(log["loss"])
    plt.plot(log["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.grid("on")
    plt.legend(["Train", "Validation"])

    # Plot accuracy on training and validation set
    plt.subplot(1, 2, 2)
    plt.plot(log["accuracy"])
    plt.plot(log["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.grid("on")
    plt.legend(["Train", "Validation"])

    # Export figure if a path is provided
    if save_path:
        plt.savefig(save_path)
    plt.show()
