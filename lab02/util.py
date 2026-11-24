import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

def _eval_split(model, x, y, criterion, device, batch_size=256):
  """Computes loss, accuracy, and AU-ROC (if binary) over a data split."""
  if len(x) == 0 or len(y) == 0:
    return None, None, None

  model.eval()
  dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y))
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

  total_loss = 0.0
  total_correct = 0
  total_samples = 0
  all_probs = []
  all_targets = []

  with torch.no_grad():
    for inputs, targets in loader:
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = model(inputs)
      loss = criterion(outputs, targets)

      total_loss += loss.item() * inputs.size(0)
      preds = outputs.argmax(dim=1)
      total_correct += (preds == targets).sum().item()
      total_samples += inputs.size(0)

      # Extract class 1 probabilities for AU-ROC
      probs = torch.softmax(outputs, dim=1)[:, 1]
      all_probs.append(probs.cpu().numpy())
      all_targets.append(targets.cpu().numpy())

  avg_loss = total_loss / total_samples
  acc = total_correct / total_samples

  # Calculate AU-ROC if binary classification
  all_probs = np.concatenate(all_probs)
  all_targets = np.concatenate(all_targets)
  unique_classes = np.unique(all_targets)

  auc = None
  if len(unique_classes) == 2:
    try:
      auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
      auc = None

  return avg_loss, acc, auc
    
def evaluate(
    model, dataset, final=False, criterion=None, device=None, batch_size=256, auc=False
):
  """Evaluates model performance on train and validation/test splits."""
  if device is None:
    device = next(model.parameters()).device
  if criterion is None:
    criterion = nn.CrossEntropyLoss()

  print('Model performance:')

  # Training set
  train_loss, train_acc, tr_auc = _eval_split(
      model,
      dataset.x_train,
      dataset.y_train,
      criterion,
      device,
      batch_size=batch_size,
  )
  print(f'\tTrain loss:          {train_loss:0.4f}')
  print(f'\tTrain accuracy:      {100 * train_acc:0.2f}%')
  if auc and tr_auc is not None:
    print(f'\tTrain AUC:           {tr_auc:0.4f}')

  score = [train_loss, train_acc]

  if final:
    test_loss, test_acc, te_auc = _eval_split(
        model,
        dataset.x_test,
        dataset.y_test,
        criterion,
        device,
        batch_size=batch_size,
    )
    if test_loss is not None:
      print(f'\tTest loss:           {test_loss:0.4f}')
      print(f'\tTest accuracy:       {100 * test_acc:0.2f}%')
      if auc and te_auc is not None:
        print(f'\tTest AUC:            {te_auc:0.4f}')
      score.extend([test_loss, test_acc])
  else:
    val_loss, val_acc, val_auc = _eval_split(
        model,
        dataset.x_valid,
        dataset.y_valid,
        criterion,
        device,
        batch_size=batch_size,
    )
    print(f'\tValidation loss:     {val_loss:0.4f}')
    print(f'\tValidation accuracy: {100 * val_acc:0.2f}%')
    if auc and val_auc is not None:
      print(f'\tValidation AUC:      {val_auc:0.4f}')
    score.extend([val_loss, val_acc])

  return score
    
# Predictions on a test set without labels, exporting
# the results to a CSV file
def pred_test(model, dataset, name, device=None, batch_size=256):
  if device is None:
    device = next(model.parameters()).device

  model.eval()

  # Create a DataLoader for the unlabelled test data
  x_test_tensor = torch.from_numpy(dataset.x_test).float()
  test_loader = DataLoader(
      TensorDataset(x_test_tensor), batch_size=batch_size, shuffle=False
  )

  preds = []
  with torch.no_grad():
    for (inputs,) in test_loader:
      inputs = inputs.to(device)
      logits = model(inputs)
      batch_pred = torch.argmax(logits, dim=1)
      preds.append(batch_pred.cpu().numpy())

  pred = np.concatenate(preds)

  df = pd.DataFrame({'class': pred})
  df.index.name = 'id'
  df.to_csv(name)

  print('Done! Please upload your file to Kaggle!')
  return pred

# Plotting of training history
def plot_training(log):
  """Plot training and validation history curves."""
  history = log['history']
  epochs = range(1, len(history['loss']) + 1)

  plt.figure(figsize=(16, 4))

  plt.subplot(1, 2, 1)
  plt.plot(epochs, history['loss'], label='Train')
  plt.plot(epochs, history['val_loss'], label='Validation')
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.grid(True)
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(epochs, history['accuracy'], label='Train')
  plt.plot(epochs, history['val_accuracy'], label='Validation')
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.grid(True)
  plt.legend()

  plt.show()


def get_weights(model):
  """Extracts weight matrices and bias vectors directly as NumPy arrays in native PyTorch format.

  Conv2d weights remain: (C_out, C_in, kH, kW) Linear weights remain:
  (out_features, in_features)
  """
  W = []
  b = []
  lname = []

  for layer in model:
    if isinstance(layer, nn.Conv2d):
      W.append(layer.weight.detach().cpu().numpy())  # (C_out, C_in, kH, kW)
      b.append(
          layer.bias.detach().cpu().numpy()
          if layer.bias is not None
          else np.zeros(layer.out_channels)
      )
      lname.append('conv')

    elif isinstance(layer, nn.MaxPool2d):
      W.append(None)
      b.append(None)
      lname.append('pool')

    elif isinstance(layer, nn.Flatten):
      W.append(None)
      b.append(None)
      lname.append('flatten')

    elif isinstance(layer, nn.Linear):
      W.append(layer.weight.detach().cpu().numpy())  # (out_features, in_features)
      b.append(
          layer.bias.detach().cpu().numpy()
          if layer.bias is not None
          else np.zeros(layer.out_features)
      )
      lname.append('dense')
    else: 
      W.append(None)
      b.append(None)
      lname.append('')        

  return W, b, lname
