import torchvision.models as models
from neural_network import NeuralNetwork
import torch
import sys
import pickle
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
sys.path.append(str(CURRENT_DIR))

from neural_network import NeuralNetwork

INTENT_TO_ID = {"GREETING": 0, "TRADE": 1, "INFO": 2, "HOSTILE": 3, "UNKNOWN": 4}
ID_TO_INTENT = {v: k for k, v in INTENT_TO_ID.items()}

VEC_PATH = ROOT_DIR / "data" / "tfidf_vectorizer.pkl"
NN_MODEL_PATH = CURRENT_DIR / "model.pth"

def load_vectorizer(path=VEC_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)

def LoadModel(path=NN_MODEL_PATH):
    model =  NeuralNetwork()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model

def Predict(model, X):
    with torch.no_grad():
        return model(X)

def text_to_vector(text: str, vectorizer, for_pytorch=True):
    sparse_vec = vectorizer.transform([text])
    if for_pytorch:
        return torch.FloatTensor(sparse_vec.toarray())
    return sparse_vec

def output_to_text(model_output, is_pytorch_logits=True) -> str:
    if is_pytorch_logits:
        pred_id = int(torch.argmax(model_output, dim=1).item())
    else:
        pred_id = int(model_output[0])
    return ID_TO_INTENT.get(pred_id, "UNKNOWN")
