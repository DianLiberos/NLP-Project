import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from network.neural_network import NeuralNetwork

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TEST_JSON_PATH = ROOT_DIR / "data" / "test.json"
MODELS_DIR = ROOT_DIR / "network" / "models"

INTENT_NAMES = ["GREETING", "TRADE", "INFO", "HOSTILE", "UNKNOWN"]
INTENT_TO_ID = {name: i for i, name in enumerate(INTENT_NAMES)}
ID_TO_INTENT = {i: name for i, name in enumerate(INTENT_NAMES)}


def get_file_size_kb(filepath: Path) -> float:
    """Trả về dung lượng file theo KB"""
    if filepath.exists():
        return round(filepath.stat().st_size / 1024, 2)
    return 0.0


def load_all_models():
    """Nạp tất cả 5 mô hình đã huấn luyện trong đồ án"""
    import pickle

    models = {}

    # 1. Multinomial Naive Bayes
    nb_path = PROCESSED_DIR / "nb_model.pkl"
    if nb_path.exists():
        with open(nb_path, "rb") as f:
            models["Multinomial NB"] = {
                "type": "sklearn",
                "model": pickle.load(f),
                "path": nb_path,
                "framework": "Scikit-Learn"
            }

    # 2. Linear SVM
    svm_path = PROCESSED_DIR / "svm_model.pkl"
    if svm_path.exists():
        with open(svm_path, "rb") as f:
            models["Linear SVM"] = {
                "type": "sklearn",
                "model": pickle.load(f),
                "path": svm_path,
                "framework": "Scikit-Learn"
            }

    # 3. Logistic Regression
    lr_path = PROCESSED_DIR / "lr_model.pkl"
    if lr_path.exists():
        with open(lr_path, "rb") as f:
            models["Logistic Regression"] = {
                "type": "sklearn",
                "model": pickle.load(f),
                "path": lr_path,
                "framework": "Scikit-Learn"
            }

    # 4. PyTorch MLP (ReLU)
    relu_path = MODELS_DIR / "ReLU" / "ReLU.pth"
    if relu_path.exists():
        nn_relu = NeuralNetwork(activation_function="ReLU")
        nn_relu.load_state_dict(torch.load(relu_path, map_location="cpu", weights_only=True))
        nn_relu.eval()
        models["PyTorch MLP (ReLU)"] = {
            "type": "pytorch",
            "model": nn_relu,
            "path": relu_path,
            "framework": "PyTorch (CPU)"
        }

    # 5. PyTorch MLP (SiLU)
    silu_path = MODELS_DIR / "SiLU" / "SiLU.pth"
    if silu_path.exists():
        nn_silu = NeuralNetwork(activation_function="SiLU")
        nn_silu.load_state_dict(torch.load(silu_path, map_location="cpu", weights_only=True))
        nn_silu.eval()
        models["PyTorch MLP (SiLU)"] = {
            "type": "pytorch",
            "model": nn_silu,
            "path": silu_path,
            "framework": "PyTorch (CPU)"
        }

    return models


def run_benchmark(save_cache: bool = True):
    """
    Chạy toàn bộ bài test đối sánh hiệu năng trên tập Test 2.000 mẫu:
    - Accuracy (%)
    - Macro F1 & Weighted F1 (%)
    - Latency (ms / query)
    - Dung lượng Model (KB/MB)
    - Confusion Matrix & Error Analysis
    """
    print("⚡ [Benchmark Suite] Đang nạp tập dữ liệu kiểm thử từ data/processed/ ...")
    X_test_sparse = sp.load_npz(PROCESSED_DIR / "X_test.npz")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    num_samples = len(y_test)

    # Đọc text gốc để phục vụ phân tích lỗi (Error Analysis)
    test_texts = []
    if TEST_JSON_PATH.exists():
        with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
            test_json = json.load(f)
            test_texts = [item["text"] for item in test_json]

    models = load_all_models()
    print(f"  ✔ Đã nạp {len(models)} mô hình.")
    print(f"  ✔ Số lượng mẫu kiểm thử: {num_samples:,} mẫu\n")

    summary_table = []
    detailed_reports = {}
    confusion_matrices = {}
    error_samples_by_model = {}

    X_test_dense_tensor = torch.FloatTensor(X_test_sparse.toarray())

    for model_name, info in models.items():
        print(f"🔄 Đang đo đạc hiệu năng: {model_name} ...")
        model = info["model"]
        mtype = info["type"]
        file_size_kb = get_file_size_kb(info["path"])

        # 1. Warm-up
        if mtype == "sklearn":
            _ = model.predict(X_test_sparse[:50])
        else:
            with torch.no_grad():
                _ = model(X_test_dense_tensor[:50])

        # 2. Đo thời gian suy luận (Latency) trên toàn bộ 2.000 mẫu
        N_RUNS = 3
        total_time_ms = 0.0
        y_preds = None

        for _ in range(N_RUNS):
            start = time.perf_counter()
            if mtype == "sklearn":
                y_preds = model.predict(X_test_sparse)
            else:
                with torch.no_grad():
                    logits = model(X_test_dense_tensor)
                    y_preds = logits.argmax(dim=1).cpu().numpy()
            end = time.perf_counter()
            total_time_ms += (end - start) * 1000

        avg_latency_per_query = total_time_ms / (N_RUNS * num_samples)

        # 3. Tính toán các chỉ số học thuật
        acc = accuracy_score(y_test, y_preds) * 100
        macro_f1 = f1_score(y_test, y_preds, average="macro") * 100
        weighted_f1 = f1_score(y_test, y_preds, average="weighted") * 100
        cm = confusion_matrix(y_test, y_preds)
        report = classification_report(
            y_test, y_preds, target_names=INTENT_NAMES, digits=4, output_dict=True
        )

        # 4. Trích xuất các ca đoán sai (Error Analysis)
        errors = []
        if test_texts:
            for idx, (true_label, pred_label) in enumerate(zip(y_test, y_preds)):
                if true_label != pred_label:
                    errors.append({
                        "text": test_texts[idx],
                        "true_intent": ID_TO_INTENT[true_label],
                        "pred_intent": ID_TO_INTENT[pred_label]
                    })
        error_samples_by_model[model_name] = errors[:8]

        summary_table.append({
            "model_name": model_name,
            "framework": info["framework"],
            "accuracy": round(acc, 2),
            "macro_f1": round(macro_f1, 2),
            "weighted_f1": round(weighted_f1, 2),
            "latency_ms": round(avg_latency_per_query, 4),
            "size_kb": file_size_kb,
            "error_count": len(errors)
        })

        detailed_reports[model_name] = report
        confusion_matrices[model_name] = cm.tolist()

    # Sắp xếp theo Accuracy giảm dần
    summary_table.sort(key=lambda x: x["accuracy"], reverse=True)

    benchmark_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_samples": num_samples,
        "summary": summary_table,
        "detailed_reports": detailed_reports,
        "confusion_matrices": confusion_matrices,
        "error_samples": error_samples_by_model
    }

    if save_cache:
        cache_path = PROCESSED_DIR / "benchmark_results.json"
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Đã lưu kết quả benchmark vào: {cache_path}")

    return benchmark_data


def print_cli_report(benchmark_data):
    """In giao diện báo cáo chuyên nghiệp ra terminal"""
    summary = benchmark_data["summary"]

    print("\n" + "=" * 50)
    print("BẢNG SO SÁNH HIỆU NĂNG TỔNG THỂ")
    print("=" * 50)
    print(
        f"{'Hạng':<5} | {'Mô hình':<24} | {'Framework':<15} | "
        f"{'Accuracy':<10} | {'Macro F1':<10} | {'Latency':<11} | {'Dung lượng':<10}"
    )
    print("-" * 88)

    rank_icons = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    for idx, row in enumerate(summary):
        icon = rank_icons[idx] if idx < len(rank_icons) else f"{idx+1}"
        size_str = (
            f"{row['size_kb']/1024:.2f} MB" if row["size_kb"] > 1024 else f"{row['size_kb']} KB"
        )
        print(
            f"{icon:<5} | {row['model_name']:<24} | {row['framework']:<15} | "
            f"{row['accuracy']:>7.2f}%  | {row['macro_f1']:>7.2f}%  | "
            f"{row['latency_ms']:>6.4f} ms | {size_str:>10}"
        )
    print("=" * 88 + "\n")

    # In Confusion Matrix của model Top 1
    top_model = summary[0]["model_name"]
    cm = np.array(benchmark_data["confusion_matrices"][top_model])
    print(f"📊 MA TRẬN NHẦM LẪN (CONFUSION MATRIX) CỦA MÔ HÌNH DẪN ĐẦU [{top_model}]:")
    header = "True \\ Pred"
    print(f"{header:<12} | " + " | ".join(f"{name:<8}" for name in INTENT_NAMES))
    print("-" * 62)
    for i, row in enumerate(cm):
        row_str = " | ".join(f"{val:>8d}" for val in row)
        print(f"{INTENT_NAMES[i]:<12} | {row_str}")
    print("-" * 62 + "\n")

    # Trích xuất một số câu đoán sai tiêu biểu
    print(f"🔍 PHÂN TÍCH MẪU ĐOÁN SAI TIÊU BIỂU (ERROR ANALYSIS - {top_model}):")
    errors = benchmark_data["error_samples"].get(top_model, [])
    if not errors:
        print("  Không có mẫu nào đoán sai!")
    else:
        for idx, err in enumerate(errors[:5], 1):
            print(
                f"  {idx}. Câu: \"{err['text']}\"\n"
                f"     ➜ Thực tế: [{err['true_intent']}] | Mô hình đoán: [{err['pred_intent']}]"
            )
    print("\n" + "=" * 88 + "\n")


if __name__ == "__main__":
    results = run_benchmark(save_cache=True)
    print_cli_report(results)
