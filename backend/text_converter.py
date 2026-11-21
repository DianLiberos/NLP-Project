import sys
import time
import pickle
from pathlib import Path
import torch
import numpy as np

# 1. Khai báo Root và nhét vào sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from network.neural_network import NeuralNetwork

# 2. Bảng mapping nhãn cố định
INTENT_TO_ID = {"GREETING": 0, "TRADE": 1, "INFO": 2, "HOSTILE": 3, "UNKNOWN": 4}
ID_TO_INTENT = {v: k for k, v in INTENT_TO_ID.items()}

# 3. ĐƯỜNG DẪN TỚI CÁC FILE MODEL TRONG DATA/PROCESSED/ VÀ NETWORK/MODELS/
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

VEC_PATH = PROCESSED_DIR / "tfidf_vectorizer.pkl"
SVM_PATH = PROCESSED_DIR / "svm_model.pkl"
NB_PATH = PROCESSED_DIR / "nb_model.pkl"
LR_PATH = PROCESSED_DIR / "lr_model.pkl"

NN_PATH = ROOT_DIR / "network" / "models" / "ReLU" / "ReLU.pth"
if not NN_PATH.exists():
    available_pth = sorted((ROOT_DIR / "network" / "models").glob("*/*.pth"))
    if available_pth:
        NN_PATH = available_pth[0]

# 4. CACHE TRÊN RAM ĐỂ TRÁNH LOAD ĐĨA NHIỀU LẦN
LOADED_MODELS = {}

def init_engines():
    """Nạp toàn bộ model lên RAM một lần duy nhất lúc khởi động app"""
    global LOADED_MODELS
    if "vectorizer" not in LOADED_MODELS and VEC_PATH.exists():
        print(f"⚡ [Backend] Đang nạp TF-IDF Vectorizer từ {VEC_PATH.name}...")
        with open(VEC_PATH, "rb") as f:
            LOADED_MODELS["vectorizer"] = pickle.load(f)

    if "svm" not in LOADED_MODELS and SVM_PATH.exists():
        print(f"⚡ [Backend] Đang nạp Linear SVM Model từ {SVM_PATH.name}...")
        with open(SVM_PATH, "rb") as f:
            LOADED_MODELS["svm"] = pickle.load(f)

    if "nb" not in LOADED_MODELS and NB_PATH.exists():
        print(f"⚡ [Backend] Đang nạp Multinomial Naive Bayes từ {NB_PATH.name}...")
        with open(NB_PATH, "rb") as f:
            LOADED_MODELS["nb"] = pickle.load(f)

    if "lr" not in LOADED_MODELS and LR_PATH.exists():
        print(f"⚡ [Backend] Đang nạp Logistic Regression từ {LR_PATH.name}...")
        with open(LR_PATH, "rb") as f:
            LOADED_MODELS["lr"] = pickle.load(f)

    if "pytorch_nn" not in LOADED_MODELS and NN_PATH.exists():
        print(f"⚡ [Backend] Đang nạp PyTorch Neural Network từ {NN_PATH.name}...")
        activation_fn = "SiLU" if "silu" in str(NN_PATH).lower() else "ReLU"
        nn_model = NeuralNetwork(activation_function=activation_fn)
        nn_model.load_state_dict(torch.load(NN_PATH, map_location="cpu", weights_only=True))
        nn_model.eval()
        LOADED_MODELS["pytorch_nn"] = nn_model

def predict_intent(text: str, engine_type: str = "Linear SVM") -> dict:
    """
    Xử lý text, đo thời gian suy luận và trả về intent + latency.
    engine_type nhận vào:
      - 'Linear SVM'
      - 'Multinomial NB'
      - 'Logistic Regression'
      - 'PyTorch Neural Net'
    """
    init_engines()
    
    if "vectorizer" not in LOADED_MODELS:
        raise FileNotFoundError(f"Không tìm thấy file Vectorizer tại {VEC_PATH}")

    vec = LOADED_MODELS["vectorizer"]
    
    # BẮT ĐẦU BẤM GIỜ 
    start_time = time.perf_counter()
    
    if engine_type == "Linear SVM":
        if "svm" not in LOADED_MODELS:
            raise FileNotFoundError(f"Không tìm thấy model SVM tại {SVM_PATH}")
        sparse_vec = vec.transform([text])
        pred_id = int(LOADED_MODELS["svm"].predict(sparse_vec)[0])

    elif engine_type == "Multinomial NB":
        if "nb" not in LOADED_MODELS:
            raise FileNotFoundError(f"Không tìm thấy model Naive Bayes tại {NB_PATH}")
        sparse_vec = vec.transform([text])
        pred_id = int(LOADED_MODELS["nb"].predict(sparse_vec)[0])

    elif engine_type == "Logistic Regression":
        if "lr" not in LOADED_MODELS:
            raise FileNotFoundError(f"Không tìm thấy model Logistic Regression tại {LR_PATH}")
        sparse_vec = vec.transform([text])
        pred_id = int(LOADED_MODELS["lr"].predict(sparse_vec)[0])
        
    elif engine_type == "PyTorch Neural Net":
        if "pytorch_nn" not in LOADED_MODELS:
            raise FileNotFoundError(f"Không tìm thấy PyTorch weights tại {NN_PATH}")
        dense_vec = vec.transform([text]).toarray()
        input_tensor = torch.FloatTensor(dense_vec)
        
        with torch.no_grad():
            logits = LOADED_MODELS["pytorch_nn"](input_tensor)
            pred_id = int(torch.argmax(logits, dim=1).item())
            
    else:
        raise ValueError(
            f"Engine '{engine_type}' không hợp lệ. "
            "Chọn: 'Linear SVM', 'Multinomial NB', 'Logistic Regression', hoặc 'PyTorch Neural Net'."
        )

    # DỪNG BẤM GIỜ
    end_time = time.perf_counter()
    latency_ms = round((end_time - start_time) * 1000, 4)
    
    final_intent = ID_TO_INTENT.get(pred_id, "UNKNOWN")
    
    return {
        "text": text,
        "intent": final_intent,
        "latency_ms": latency_ms,
        "engine": engine_type
    }

if __name__ == "__main__":
    init_engines()
    test_query = "ong chu co ban kiem go khong?"
    N_RUNS = 1000

    engines = ["Linear SVM", "Multinomial NB", "Logistic Regression", "PyTorch Neural Net"]

    # 1. Warm-up
    print(f"\n🔍 Query thử nghiệm: \"{test_query}\"")
    print("⏳ Đang warm-up và đo tốc độ suy luận...")
    for eng in engines:
        for _ in range(20):
            predict_intent(test_query, engine_type=eng)

    # 2. Đo tốc độ từng Engine
    benchmarks = {}
    sample_outputs = {}
    for eng in engines:
        start = time.perf_counter()
        for _ in range(N_RUNS):
            res = predict_intent(test_query, engine_type=eng)
        elapsed_ms = (time.perf_counter() - start) * 1000
        benchmarks[eng] = elapsed_ms / N_RUNS
        sample_outputs[eng] = res["intent"]

    print("\n" + "=" * 65)
    print(f"{'Engine':<25} | {'Dự đoán Intent':<15} | {'Tốc độ (ms/query)':<18}")
    print("-" * 65)
    for eng in engines:
        print(f"{eng:<25} | {sample_outputs[eng]:<15} | {benchmarks[eng]:<18.4f}")
    print("=" * 65 + "\n")
