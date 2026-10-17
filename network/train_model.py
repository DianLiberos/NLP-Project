import sys
from pathlib import Path
from neural_network import NeuralNetwork
import numpy as np
import os 
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, dataloader

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

sys.path.append(str(CURRENT_DIR))
from neural_network import NeuralNetwork

NPZ_DIR = ROOT_DIR / "data" / "npz" / "old_npz_data"
NPY_DIR = ROOT_DIR / "data" / "npy" / "old_npy_data"
MODEL_SAVE_PATH = CURRENT_DIR / "model.pth"

X_train_dense = sp.load_npz(NPZ_DIR / "X_train.npz").toarray()
X_test_dense = sp.load_npz(NPZ_DIR / "X_test.npz").toarray()
X_train_tensor = torch.FloatTensor(X_train_dense)
y_train_tensor = torch.LongTensor(np.load(NPY_DIR / "y_train.npy"))

X_test_tensor = torch.FloatTensor(X_test_dense)
y_test_tensor = torch.LongTensor(np.load(NPY_DIR / "y_test.npy"))

train_dataloader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
test_dataloader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        #compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        #backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * dataloader.batch_size + len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
torch.save(model.state_dict(), "network/model_test.pth")
print("Saved model")
