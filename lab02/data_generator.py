import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

class DataGenerator:

  def __init__(self, verbose=True):
    self.verbose = verbose

  def generate(
      self,
      dataset='mnist',  # Dataset type: 'mnist', 'cifar10', or 'patchcam'
      N_train=None,  # Number of training samples (None uses all available)
      N_valid=0.1,  # Fraction of training samples to use as validation data
      download_dir='./data',
  ):

    self.N_train = N_train
    self.N_valid = N_valid
    self.dataset = dataset.lower()

    if self.dataset == 'mnist':
      train_set = datasets.MNIST(
          root=download_dir, train=True, download=True
      )
      test_set = datasets.MNIST(
          root=download_dir, train=False, download=True
      )

      # Shape: (N, 28, 28) -> uint8
      self.x_train = train_set.data.numpy()
      self.y_train = train_set.targets.numpy()
      self.x_test = test_set.data.numpy()
      self.y_test = test_set.targets.numpy()

      self.split_data()
      self.normalize()

      # Add channel dimension: (N, 28, 28) -> (N, 1, 28, 28) [NCHW]
      self.x_train = np.expand_dims(self.x_train, 1)
      self.x_valid = np.expand_dims(self.x_valid, 1)
      self.x_test = np.expand_dims(self.x_test, 1)

    elif self.dataset == 'cifar10':
      train_set = datasets.CIFAR10(
          root=download_dir, train=True, download=True
      )
      test_set = datasets.CIFAR10(
          root=download_dir, train=False, download=True
      )

      # Shape: (N, 32, 32, 3) -> uint8
      self.x_train = train_set.data
      self.y_train = np.array(train_set.targets)
      self.x_test = test_set.data
      self.y_test = np.array(test_set.targets)

      self.split_data()
      self.normalize()

      # Transpose from NHWC to NCHW: (N, H, W, C) -> (N, C, H, W)
      self.x_train = np.transpose(self.x_train, (0, 3, 1, 2))
      self.x_valid = np.transpose(self.x_valid, (0, 3, 1, 2))
      self.x_test = np.transpose(self.x_test, (0, 3, 1, 2))
    
    elif self.dataset == 'patchcam':
      with h5py.File('patchcam/train.h5', 'r') as f:
        self.x_train = f['x'][:]
        self.y_train = f['y'][:].squeeze()
      with h5py.File('patchcam/valid.h5', 'r') as f:
        self.x_valid = f['x'][:]
        self.y_valid = f['y'][:].squeeze()
      with h5py.File('patchcam/test_x.h5', 'r') as f:
        self.x_test = f['x'][:]
        self.y_test = np.array([], dtype=np.int64)

      self.normalize()

      # Convert NHWC -> NCHW
      self.x_train = np.transpose(self.x_train, (0, 3, 1, 2))
      self.x_valid = np.transpose(self.x_valid, (0, 3, 1, 2))
      self.x_test = np.transpose(self.x_test, (0, 3, 1, 2))

    else:
      raise ValueError(f'Unknown dataset: {dataset}')

    print('arrived here')

    # Metadata
    self.K = len(np.unique(self.y_train))
    self.C = self.x_train.shape[1]  # Channel is at index 1 in NCHW

    # Ensure labels are 1D int64 arrays (required by PyTorch cross-entropy)
    self.y_train = self.y_train.astype(np.int64).reshape(-1)
    self.y_valid = self.y_valid.astype(np.int64).reshape(-1)
    if len(self.y_test) > 0:
      self.y_test = self.y_test.astype(np.int64).reshape(-1)

    if self.verbose:
      print('Data specification:')
      print(f'\tDataset type:          {self.dataset}')
      #print(f'\tNumber of classes (K): {self.K}')
      print(f'\tNumber of channels (C): {self.C}')
      print(f'\tTraining data shape:   {self.x_train.shape} [NCHW]')
      print(f'\tValidation data shape: {self.x_valid.shape} [NCHW]')
      print(f'\tTest data shape:       {self.x_test.shape} [NCHW]')


  def split_data(self):
    N = self.x_train.shape[0]
    indices = np.random.permutation(N)
    self.x_train = self.x_train[indices]
    self.y_train = self.y_train[indices]

    n_val = int(N * self.N_valid)
    n_train = N - n_val

    self.x_valid = self.x_train[-n_val:]
    self.y_valid = self.y_train[-n_val:]

    if self.N_train and self.N_train < n_train:
      self.x_train = self.x_train[: self.N_train]
      self.y_train = self.y_train[: self.N_train]
    else:
      self.x_train = self.x_train[:n_train]
      self.y_train = self.y_train[:n_train]
      self.N_train = n_train

  def normalize(self):
    # Scale from [0, 255] to [-1.0, 1.0] as float32
    self.x_train = (2.0 * (self.x_train.astype(np.float32) / 255.0)) - 1.0
    if hasattr(self, 'x_valid') and self.x_valid is not None:
      self.x_valid = (2.0 * (self.x_valid.astype(np.float32) / 255.0)) - 1.0
    self.x_test = (2.0 * (self.x_test.astype(np.float32) / 255.0)) - 1.0

  def get_dataset(self, split='train'):
    """Returns a PyTorch TensorDataset for a given split."""
    if split == 'train':
      return TensorDataset(
          torch.from_numpy(self.x_train), torch.from_numpy(self.y_train)
      )
    elif split == 'valid':
      return TensorDataset(
          torch.from_numpy(self.x_valid), torch.from_numpy(self.y_valid)
      )
    elif split == 'test':
      targets = (
          torch.from_numpy(self.y_test)
          if len(self.y_test) > 0
          else torch.empty(0)
      )
      return TensorDataset(torch.from_numpy(self.x_test), targets)
    else:
      raise ValueError(f'Invalid split: {split}')

  def get_loader(self, split='train', batch_size=64, shuffle=None):
    """Convenience method to retrieve standard PyTorch DataLoaders."""
    ds = self.get_dataset(split)
    if shuffle is None:
      shuffle = split == 'train'
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

  def plot(self, xx=12, yy=3, save_path=None):
    plt.figure(figsize=(18, yy * 2))
    cm = 'gray' if self.C == 1 else None

    for i in range(xx * yy):
      if i >= len(self.x_train):
        break
      plt.subplot(yy, xx, i + 1)

      # Convert NCHW back to HWC or HW for matplotlib
      img = (self.x_train[i] + 1.0) / 2.0
      if self.C == 1:
        img_disp = img.squeeze(0)  # (1, H, W) -> (H, W)
      else:
        img_disp = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)

      plt.imshow(img_disp, cmap=cm)
      plt.title(f'label={self.y_train[i]}')
      plt.axis('off')

    plt.tight_layout()
    if save_path:
      plt.savefig(save_path)
    plt.show()







