import pickle
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

INTENT_NAMES = ["GREETING", "TRADE", "INFO", "HOSTILE", "UNKNOWN"]

def main():
    print("⚡ [Train ML] Đang nạp dữ liệu từ data/processed/ ...")
    X_train = sp.load_npz(PROCESSED_DIR / "X_train.npz")
    X_test = sp.load_npz(PROCESSED_DIR / "X_test.npz")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    print(f"  ✔ X_train shape: {X_train.shape}, y_train: {len(y_train)}")
    print(f"  ✔ X_test shape:  {X_test.shape}, y_test:  {len(y_test)}\n")

    models = {
        "MultinomialNB": MultinomialNB(),
        "LinearSVC (C=0.1)": LinearSVC(C=0.1, random_state=42, dual=False),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    }

    trained_models = {}
    results = []

    for name, model in models.items():
        print(f"🔄 Đang huấn luyện {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred) * 100
        f1_macro = f1_score(y_test, y_pred, average="macro") * 100
        results.append((name, acc, f1_macro))
        trained_models[name] = (model, y_pred)

    print("\n" + "=" * 58)
    print(f"{'Mô hình':<25} | {'Accuracy (%)':<14} | {'Macro F1 (%)':<12}")
    print("-" * 58)
    for name, acc, f1 in results:
        print(f"{name:<25} | {acc:<14.2f} | {f1:<12.2f}")
    print("=" * 58 + "\n")

    # In chi tiết báo cáo phân loại của MultinomialNB (Model dẫn đầu)
    nb_model, nb_pred = trained_models["MultinomialNB"]
    print("📊 Báo cáo phân loại chi tiết của MultinomialNB:")
    print(classification_report(y_test, nb_pred, target_names=INTENT_NAMES, digits=4))

    # In chi tiết báo cáo phân loại của LinearSVC
    svm_model, svm_pred = trained_models["LinearSVC (C=0.1)"]
    print("📊 Báo cáo phân loại chi tiết của LinearSVC (C=0.1):")
    print(classification_report(y_test, svm_pred, target_names=INTENT_NAMES, digits=4))

    # Lưu đầy đủ 3 mô hình
    save_map = {
        "svm_model.pkl": svm_model,
        "nb_model.pkl": nb_model,
        "lr_model.pkl": trained_models["LogisticRegression"][0],
    }

    print("💾 Đang lưu các mô hình vào data/processed/ ...")
    for filename, model_obj in save_map.items():
        save_path = PROCESSED_DIR / filename
        with open(save_path, "wb") as f:
            pickle.dump(model_obj, f)
        print(f"   ✔ Đã lưu: {save_path.name}")

    print("\n✅ Hoàn tất huấn luyện và lưu trữ 3 mô hình!")

if __name__ == "__main__":
    main()
