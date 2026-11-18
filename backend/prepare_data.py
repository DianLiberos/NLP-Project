import json
import pickle
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_PATH = DATA_DIR / "train.json"
TEST_PATH = DATA_DIR / "test.json"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INTENT_TO_ID = {"GREETING": 0, "TRADE": 1, "INFO": 2, "HOSTILE": 3, "UNKNOWN": 4}

def main():
    print("🚀 [Prepare Data] Đang đọc trực tiếp train.json và test.json (Zero-Leakage Pipeline)...")
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"  ✔ Tập Train: {len(train_data):,} mẫu")
    print(f"  ✔ Tập Test:  {len(test_data):,} mẫu")

    X_train_texts = [item["text"] for item in train_data]
    y_train = np.array([INTENT_TO_ID[item["intent"]] for item in train_data])

    X_test_texts = [item["text"] for item in test_data]
    y_test = np.array([INTENT_TO_ID[item["intent"]] for item in test_data])

    # Vector hóa TF-IDF: Chỉ fit trên Train, transform trên Test
    print("⚙️ Đang vector hóa TF-IDF (max_features=2000, ngram_range=(1,2), sublinear_tf=True)...")
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_sparse = vectorizer.fit_transform(X_train_texts)
    X_test_sparse = vectorizer.transform(X_test_texts)

    print(f"📦 Đang xuất dữ liệu ra thư mục: {PROCESSED_DIR} ...")
    with open(PROCESSED_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    sp.save_npz(PROCESSED_DIR / "X_train.npz", X_train_sparse)
    sp.save_npz(PROCESSED_DIR / "X_test.npz", X_test_sparse)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)

    print("✅ Hoàn tất xuất ma trận Sparse và nhãn vào data/processed/:")
    print(f"   🔹 X_train shape: {X_train_sparse.shape} | y_train: {y_train.shape}")
    print(f"   🔸 X_test shape:  {X_test_sparse.shape} | y_test:  {y_test.shape}")

if __name__ == "__main__":
    main()
