import pickle
import torch
import numpy as np
from network.neural_network import NeuralNetwork

# 1. Bảng map cố định giữa Chữ (Intent) và Số (ID)
INTENT_TO_ID = {"GREETING": 0, "TRADE": 1, "INFO": 2, "HOSTILE": 3, "UNKNOWN": 4}
ID_TO_INTENT = {v: k for k, v in INTENT_TO_ID.items()}

# 2. Load file dịch TF-IDF 
def load_vectorizer(vec_path="data/tfidf_vectorizer.pkl"):
    with open(vec_path, "rb") as f:
        return pickle.load(f)

# 3. Đổi INPUT: Từ Chữ (String) -> Số (PyTorch Tensor hoặc Sparse Matrix)
def text_to_vector(text: str, vectorizer, for_pytorch=True):
    # Ép câu chữ thành ma trận thưa (1 x 2000)
    sparse_vec = vectorizer.transform([text])
    
    if for_pytorch:
        # Mạng Nơ-ron của thằng bạn bắt buộc phải ăn Dense Tensor (1, 2000)
        dense_vec = sparse_vec.toarray()
        return torch.FloatTensor(dense_vec)
    
    return sparse_vec

# 4. Đổi OUTPUT: Từ Số (Logits của NN hoặc ID của SVM) -> Chữ (Intent String)
def output_to_text(model_output, is_pytorch_logits=True) -> str:
    if is_pytorch_logits:
        # model_output đang là mảng 5 số thực (logits), lấy vị trí có số to nhất
        pred_id = int(torch.argmax(model_output, dim=1).item())
    else:
        # Nếu là output của SVM thì nó là số nguyên sẵn rồi
        pred_id = int(model_output[0])
        
    return ID_TO_INTENT.get(pred_id, "UNKNOWN")

if __name__ == "__main__":
    vec = load_vectorizer("data/tfidf_vectorizer.pkl")
    
    nn_model = NeuralNetwork()
    nn_model.load_state_dict(torch.load("network/model.pth", weights_only=True))
    nn_model.eval()
    
    # Giả lập người chơi nhập chat
    sample_text = "dm shop ban dat vcl"
    
    # Bước 1: Chữ -> Số (Tensor)
    input_tensor = text_to_vector(sample_text, vec, for_pytorch=True)
    
    # Bước 2: Đưa vào Mạng Nơ-ron
    with torch.no_grad():
        raw_logits = nn_model(input_tensor)
        
    # Bước 3: Số (Logits) -> Chữ (Nhãn intent)
    final_label = output_to_text(raw_logits, is_pytorch_logits=True)
    
    print(f"User gõ: '{sample_text}'")
    print(f"Logits thô từ NN: {raw_logits.numpy()}")
    print(f"=> Chốt nhãn chữ: {final_label}")