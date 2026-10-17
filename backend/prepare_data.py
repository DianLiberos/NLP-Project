import json
import pickle
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
NPZ_DIR = DATA_DIR / "npz" / "old_npz_data"
NPY_DIR = DATA_DIR / "npy" / "old_npy_data"
NPZ_DIR.mkdir(parents=True, exist_ok=True)
NPY_DIR.mkdir(parents=True, exist_ok=True)

# Tạo folder data/ nếu chưa có
os.makedirs('data', exist_ok=True)

# 1. Load Data gốc
print("Đang đọc dataset gốc...")
with open('data/train_augmented.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = [item['text'] for item in data]
intent_to_id = {"GREETING": 0, "TRADE": 1, "INFO": 2, "HOSTILE": 3, "UNKNOWN": 4}
y = [intent_to_id[item['intent']] for item in data]

# 2. Chia cố định Train/Test (80/20)
# X là mảng chứa các câu thoại
# y là mảng chứa các nhãn tương ứng đã gán
print("Đang chia tập Train/Test...")
X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Vector hóa bằng TF-IDF
print("Đang ép kiểu TF-IDF (Sparse Matrix)...")
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X_train_sparse = vectorizer.fit_transform(X_train_text)
X_test_sparse = vectorizer.transform(X_test_text)

print("Đang xuất file vào folder data/ ...")

# Lưu Từ điển Vectorizer (cho backend FastAPI sau này)
with open('data/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Lưu ma trận thưa X (Sparse Matrix) thành định dạng .npz
sp.save_npz('data/X_train.npz', X_train_sparse)
sp.save_npz('data/X_test.npz', X_test_sparse)

# Lưu mảng nhãn y thành định dạng .npy
np.save('data/y_train.npy', np.array(y_train))
np.save('data/y_test.npy', np.array(y_test))

print("✅ Đã lưu vào folder data")