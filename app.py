import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import packaging.version
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# Fix lỗi tương thích torch packaging vendor trên Windows / Python 3.13
sys.modules.setdefault("torch._vendor.packaging.version", packaging.version)
import torch

BASE_DIR = Path(__file__).resolve().parent
NETWORK_DIR = BASE_DIR / "network"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = NETWORK_DIR / "models"
BENCHMARK_CACHE_PATH = PROCESSED_DIR / "benchmark_results.json"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(NETWORK_DIR) not in sys.path:
    sys.path.insert(0, str(NETWORK_DIR))

from network.neural_network import NeuralNetwork

st.set_page_config(
    page_title="Hệ thống Nhận diện Ý định NPC",
    layout="wide"
)

INTENT_NAMES = ["GREETING", "TRADE", "INFO", "HOSTILE", "UNKNOWN"]
ID_TO_INTENT = {i: name for i, name in enumerate(INTENT_NAMES)}


@st.cache_resource
def load_all_inference_engines():
    """Nạp sẵn từ điển vectorizer và toàn bộ mô hình lên RAM để suy luận tức thì"""
    engines = {}

    vec_path = PROCESSED_DIR / "tfidf_vectorizer.pkl"
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)

    # 1. Multinomial Naive Bayes
    nb_path = PROCESSED_DIR / "nb_model.pkl"
    if nb_path.exists():
        with open(nb_path, "rb") as f:
            engines["Multinomial Naive Bayes"] = {"type": "sklearn", "model": pickle.load(f)}

    # 2. PyTorch MLP (ReLU)
    relu_path = MODELS_DIR / "ReLU" / "ReLU.pth"
    if relu_path.exists():
        nn_relu = NeuralNetwork(activation_function="ReLU")
        nn_relu.load_state_dict(torch.load(relu_path, map_location="cpu", weights_only=True))
        nn_relu.eval()
        engines["PyTorch MLP (ReLU)"] = {"type": "pytorch", "model": nn_relu}

    # 3. PyTorch MLP (SiLU)
    silu_path = MODELS_DIR / "SiLU" / "SiLU.pth"
    if silu_path.exists():
        nn_silu = NeuralNetwork(activation_function="SiLU")
        nn_silu.load_state_dict(torch.load(silu_path, map_location="cpu", weights_only=True))
        nn_silu.eval()
        engines["PyTorch MLP (SiLU)"] = {"type": "pytorch", "model": nn_silu}

    # 4. Linear SVM
    svm_path = PROCESSED_DIR / "svm_model.pkl"
    if svm_path.exists():
        with open(svm_path, "rb") as f:
            engines["Linear SVM"] = {"type": "sklearn", "model": pickle.load(f)}

    # 5. Logistic Regression
    lr_path = PROCESSED_DIR / "lr_model.pkl"
    if lr_path.exists():
        with open(lr_path, "rb") as f:
            engines["Logistic Regression"] = {"type": "sklearn", "model": pickle.load(f)}

    return vectorizer, engines


@st.cache_data
def load_benchmark_data():
    """Đọc kết quả kiểm thử từ file cache JSON đã chạy trước đó"""
    if not BENCHMARK_CACHE_PATH.exists():
        from backend.benchmark import run_benchmark
        return run_benchmark(save_cache=True)
    with open(BENCHMARK_CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_text(text: str, engine_name: str, vectorizer, engines):
    """Thực hiện suy luận và đo độ trễ mili-giây"""
    info = engines[engine_name]
    mtype = info["type"]
    model = info["model"]

    start = time.perf_counter()
    sparse_vec = vectorizer.transform([text])

    if mtype == "sklearn":
        pred_id = int(model.predict(sparse_vec)[0])
    else:
        dense_tensor = torch.FloatTensor(sparse_vec.toarray())
        with torch.no_grad():
            logits = model(dense_tensor)
            pred_id = int(logits.argmax(dim=1).item())

    latency_ms = (time.perf_counter() - start) * 1000
    predicted_intent = ID_TO_INTENT.get(pred_id, "UNKNOWN")
    return predicted_intent, latency_ms


vectorizer, engines = load_all_inference_engines()

# Tiêu đề giao diện (tuyệt đối không sử dụng icon)
st.title("Hệ thống Nhận diện Ý định NPC")

# Tạo 2 Tab chính (tuyệt đối không dùng icon ở tiêu đề)
tab_chat, tab_benchmark = st.tabs(["Trò chuyện NPC", "Hiệu năng và Đối sánh"])

# ==============================================================================
# TAB 1: TRÒ CHUYỆN VỚI NPC
# ==============================================================================
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Thanh điều khiển tùy chọn mô hình
    col_sel, col_info = st.columns([3, 2])
    with col_sel:
        selected_model_name = st.selectbox(
            label="Chọn mô hình suy luận:",
            options=list(engines.keys()),
            index=0
        )
    with col_info:
        st.caption(
            f"Mô hình đang kích hoạt: **{selected_model_name}** | "
            f"Không gian đặc trưng: TF-IDF 2.000 chiều"
        )

    st.markdown("---")

    # Hiển thị lịch sử trò chuyện
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "meta" in msg:
                    st.caption(msg["meta"])

    # Ô nhập liệu chat
    user_input = st.chat_input("Nhập tin nhắn gửi tới NPC...")

    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        pred_intent, latency = predict_text(user_input, selected_model_name, vectorizer, engines)
        meta_info = f"Mô hình: {selected_model_name} | Thời gian phản hồi: {latency:.2f} ms"

        st.session_state.messages.append({
            "role": "assistant",
            "content": f"**Ý định nhận diện:** `{pred_intent}`",
            "meta": meta_info
        })
        st.rerun()

# ==============================================================================
# TAB 2: BẢNG ĐIỀU KHIỂN HIỆU NĂNG VÀ ĐỐI SÁNH (BENCHMARK DASHBOARD)
# ==============================================================================
with tab_benchmark:
    st.subheader("Báo cáo đối sánh các mô hình trên tập kiểm thử độc lập")
    st.caption("Tập kiểm thử chuẩn gồm 2.000 mẫu, bao gồm 70% cấu trúc câu chưa từng thấy trong tập huấn luyện.")

    col_btn, col_time = st.columns([2, 3])
    with col_btn:
        if st.button("Chạy lại đánh giá toàn diện"):
            with st.spinner("Đang kiểm thử đồng thời 5 mô hình trên 2.000 mẫu..."):
                from backend.benchmark import run_benchmark
                st.cache_data.clear()
                benchmark_data = run_benchmark(save_cache=True)
                st.success("Đã hoàn tất kiểm thử!")
                st.rerun()
            st.cache_data.clear()
            st.rerun()

    benchmark_data = load_benchmark_data()
    with col_time:
        st.caption(f"Thời gian cập nhật kết quả: {benchmark_data.get('timestamp', 'Mới nhất')}")

    # 1. Bảng số liệu tổng hợp
    st.markdown("#### Bảng đối sánh tổng thể")
    summary_raw = benchmark_data["summary"]

    formatted_rows = []
    chart_models = []
    chart_acc = []
    chart_f1 = []

    for rank_idx, row in enumerate(summary_raw, start=1):
        size_display = (
            f"{row['size_kb']/1024:.2f} MB" if row["size_kb"] > 1024 else f"{row['size_kb']} KB"
        )
        # Sử dụng số 1, 2, 3, 4, 5 để phân hạng (tuyệt đối không sử dụng icon)
        formatted_rows.append({
            "Hạng": rank_idx,
            "Mô hình": row["model_name"],
            "Framework": row["framework"],
            "Độ chính xác (%)": f"{row['accuracy']:.2f}%",
            "Macro F1 (%)": f"{row['macro_f1']:.2f}%",
            "Weighted F1 (%)": f"{row['weighted_f1']:.2f}%",
            "Độ trễ trung bình": f"{row['latency_ms']:.4f} ms",
            "Dung lượng lưu trữ": size_display
        })

        chart_models.append(row["model_name"])
        chart_acc.append(row["accuracy"])
        chart_f1.append(row["macro_f1"])

    df_summary = pd.DataFrame(formatted_rows)
    st.dataframe(df_summary, width="stretch", hide_index=True)

    st.markdown("---")

    # 2. Biểu đồ cột so sánh Accuracy và Macro F1
    st.markdown("#### Biểu đồ so sánh Độ chính xác giữa các mô hình")
    df_chart = pd.DataFrame({
        "Mô hình": chart_models,
        "Độ chính xác (Accuracy %)": chart_acc,
        "Macro F1 (%)": chart_f1
    }).set_index("Mô hình")

    st.bar_chart(df_chart, height=350)

    st.markdown("---")

    # 3. Ma trận nhầm lẫn (Confusion Matrix)
    st.markdown("#### Ma trận nhầm lẫn chi tiết")
    model_choices = [r["model_name"] for r in summary_raw]
    selected_cm_model = st.selectbox(
        label="Chọn mô hình để xem ma trận nhầm lẫn:",
        options=model_choices,
        index=0
    )

    if selected_cm_model in benchmark_data["confusion_matrices"]:
        cm_matrix = np.array(benchmark_data["confusion_matrices"][selected_cm_model])
        df_cm = pd.DataFrame(
            cm_matrix,
            index=[f"Thực tế: {name}" for name in INTENT_NAMES],
            columns=[f"Dự đoán: {name}" for name in INTENT_NAMES]
        )
        st.dataframe(df_cm, width="stretch")

    st.markdown("---")

    # 4. Phân tích các mẫu dự đoán sai tiêu biểu (Error Analysis)
    st.markdown("#### Phân tích các trường hợp dự đoán sai tiêu biểu")
    errors = benchmark_data["error_samples"].get(selected_cm_model, [])

    if not errors:
        st.info("Mô hình này không có trường hợp nào dự đoán sai trong số mẫu được trích xuất.")
    else:
        df_errors = pd.DataFrame([
            {
                "STT": idx + 1,
                "Câu truy vấn": err["text"],
                "Ý định thực tế": err["true_intent"],
                "Ý định dự đoán": err["pred_intent"]
            }
            for idx, err in enumerate(errors)
        ])
        st.dataframe(df_errors, width="stretch", hide_index=True)

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.header("Cài đặt hệ thống")
    if st.button("Xóa lịch sử trò chuyện"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("##### Thông số dữ liệu hiện tại")
    st.markdown("- **Tập huấn luyện:** 9.215 mẫu")
    st.markdown("- **Tập kiểm thử:** 2.000 mẫu")
    st.markdown("- **Số lớp ý định:** 5 nhãn")
    st.markdown("  - `GREETING` (Chào hỏi)")
    st.markdown("  - `TRADE` (Giao dịch)")
    st.markdown("  - `INFO` (Hỏi thông tin)")
    st.markdown("  - `HOSTILE` (Khiêu khích)")
    st.markdown("  - `UNKNOWN` (Ngoại luồng)")
    st.markdown("---")
    device_name = "CUDA (NVIDIA GPU)" if torch.cuda.is_available() else "CPU"
    st.caption(f"Thiết bị phần cứng: {device_name}")
