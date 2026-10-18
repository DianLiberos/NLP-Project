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

# 3. ĐƯỜNG DẪN TỚI CÁC FILE MODEL
VEC_PATH = ROOT_DIR / "data" / "tfidf_vectorizer.pkl"
SVM_PATH = ROOT_DIR / "network" / "svm_model.pkl"
NN_PATH = ROOT_DIR / "network" / "model.pth"

# 4. CACHE TRÊN RAM ĐỂ TRÁNH LOAD ĐĨA NHIỀU LẦN
LOADED_MODELS = {}

def init_engines():
    """Nạp toàn bộ model lên RAM một lần duy nhất lúc khởi động app"""
    global LOADED_MODELS
    if "vectorizer" not in LOADED_MODELS:
        print("⚡ [Backend] Đang nạp TF-IDF Vectorizer...")
        with open(VEC_PATH, "rb") as f:
            LOADED_MODELS["vectorizer"] = pickle.load(f)

    if "svm" not in LOADED_MODELS and SVM_PATH.exists():
        print("⚡ [Backend] Đang nạp Linear SVM Model...")
        with open(SVM_PATH, "rb") as f:
            LOADED_MODELS["svm"] = pickle.load(f)

    if "pytorch_nn" not in LOADED_MODELS and NN_PATH.exists():
        print("⚡ [Backend] Đang nạp PyTorch Neural Network...")
        nn_model = NeuralNetwork()
        nn_model.load_state_dict(torch.load(NN_PATH, map_location="cpu", weights_only=True))
        nn_model.eval()
        LOADED_MODELS["pytorch_nn"] = nn_model

# 5. HÀM CHÍNH ĐỂ THẰNG SỐ 3 IMPORT VÀO APP.PY
def predict_intent(text: str, engine_type: str = "Linear SVM") -> dict:
    """
    Xử lý text, đo thời gian suy luận và trả về intent + latency.
    engine_type nhận vào: 'Linear SVM' hoặc 'PyTorch Neural Net'
    """
    # Đảm bảo model đã nằm sẵn trên RAM
    init_engines()
    
    vec = LOADED_MODELS["vectorizer"]
    
    # BẮT ĐẦU BẤM GIỜ 
    start_time = time.perf_counter()
    
    if engine_type == "Linear SVM":
        if "svm" not in LOADED_MODELS:
            raise FileNotFoundError(f"Không tìm thấy model SVM tại {SVM_PATH}")
        
        # Linear SVM dùng trực tiếp ma trận thưa (Sparse Matrix) 
        sparse_vec = vec.transform([text])
        pred_id = int(LOADED_MODELS["svm"].predict(sparse_vec)[0])
        
    elif engine_type == "PyTorch Neural Net":
        if "pytorch_nn" not in LOADED_MODELS:
            raise FileNotFoundError(f"Không tìm thấy PyTorch weights tại {NN_PATH}")
        
        # Mạng nơ-ron phải convert qua Dense Array rồi ép thành PyTorch Tensor
        dense_vec = vec.transform([text]).toarray()
        input_tensor = torch.FloatTensor(dense_vec)
        
        with torch.no_grad():
            logits = LOADED_MODELS["pytorch_nn"](input_tensor)
            pred_id = int(torch.argmax(logits, dim=1).item())
            
    else:
        raise ValueError(f"Engine '{engine_type}' không hợp lệ. Chọn 'Linear SVM' hoặc 'PyTorch Neural Net'.")

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

    # 1. Warm-up
    for _ in range(20):
        predict_intent(test_query, engine_type="Linear SVM")
        predict_intent(test_query, engine_type="PyTorch Neural Net")

    # 2. Đo Linear SVM
    start_svm = time.perf_counter()
    for _ in range(N_RUNS):
        predict_intent(test_query, engine_type="Linear SVM")
    total_svm_ms = (time.perf_counter() - start_svm) * 1000
    avg_svm = total_svm_ms / N_RUNS

    # 3. Đo PyTorch Neural Net
    start_nn = time.perf_counter()
    for _ in range(N_RUNS):
        predict_intent(test_query, engine_type="PyTorch Neural Net")
    total_nn_ms = (time.perf_counter() - start_nn) * 1000
    avg_nn = total_nn_ms / N_RUNS

    print("\n")
    print("KẾT QUẢ TỐC ĐỘ: ")
    print(f"🔹 Linear SVM:         {avg_svm:.4f} ms / query")
    print(f"🔸 PyTorch Neural Net: {avg_nn:.4f} ms / query")