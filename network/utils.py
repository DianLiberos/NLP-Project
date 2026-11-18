import sys
import pickle
from pathlib import Path
import torch

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from neural_network import NeuralNetwork

INTENT_TO_ID = {"GREETING": 0, "TRADE": 1, "INFO": 2, "HOSTILE": 3, "UNKNOWN": 4}
ID_TO_INTENT = {v: k for k, v in INTENT_TO_ID.items()}

PROCESSED_VEC_PATH = ROOT_DIR / "data" / "processed" / "tfidf_vectorizer.pkl"
LEGACY_VEC_PATH = ROOT_DIR / "data" / "tfidf_vectorizer.pkl"
VEC_PATH = PROCESSED_VEC_PATH if PROCESSED_VEC_PATH.exists() else LEGACY_VEC_PATH

_vectorizer_cache = None

def load_vectorizer(path=None, use_cache=True):
    global _vectorizer_cache

    if path is None:
        path = PROCESSED_VEC_PATH if PROCESSED_VEC_PATH.exists() else LEGACY_VEC_PATH

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
