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
_vectorizer_cache = None

def load_vectorizer(path=VEC_PATH, use_cache=True):
    global _vectorizer_cache

    if use_cache and _vectorizer_cache is not None:
        return _vectorizer_cache

    try:
        with open(path, "rb") as f:
            vectorizer = pickle.load(f)
    except pickle.UnpicklingError as e:
        raise pickle.UnpicklingError(f"Failed to load vectorizer from {path}: {e}")

    if use_cache:
        _vectorizer_cache = vectorizer
    return _vectorizer_cache

def LoadModel(path, activation_function=None):
    if activation_function is None:
        activation_function = "SiLU" if "silu" in str(path).lower() else "ReLU"
    model = NeuralNetwork(activation_function=activation_function)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model

def Predict(model, X):
    with torch.no_grad():
        return model(X)

def Text_to_vector(text: str, vectorizer=None, for_pytorch=True):
    if vectorizer is None:
        vectorizer = load_vectorizer()

    sparse_vec = vectorizer.transform([text])

    if for_pytorch:
        return torch.FloatTensor(sparse_vec.toarray())
    return sparse_vec

def Output_to_text(model_output, is_pytorch_logits=True) -> str:
    if is_pytorch_logits:
        pred_id = int(torch.argmax(model_output, dim=1).item())
    else:
        pred_id = int(model_output[0])
    return ID_TO_INTENT.get(pred_id, "UNKNOWN")

# Cach xai
# model = LoadModel(path) path la dir toi model CURRENT_DIR/models/ReLU or SiLU/example.pth (trong ReLU or SiLU co nhieu model lua mot cai)
# X = Text_to_vector('text user trong nay') chuyen text user nhap vao thanh vector
# y = Predict(model, X) bat dau doan 
# print(Output_to_text(y)) (chuyen output thanh text se la mot trong 5 nhan o day print ra cho de hieu khong nhat thiet phai print)
