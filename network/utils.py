import torchvision.models as models
from neural_network import NeuralNetwork
import torch

def LoadModel(directory):
    model =  NeuralNetwork()
    model.load_state_dict(torch.load(directory, weights_only=True))
    model.eval()
    return model

def Predict(model, X):
    with torch.no_grad():
        return model(X)
