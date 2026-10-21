import sys
from pathlib import Path
import warnings
import packaging.version
import streamlit as st

warnings.filterwarnings("ignore")

# Fix lỗi tương thích torch packaging vendor trên Windows / Python 3.13
sys.modules.setdefault("torch._vendor.packaging.version", packaging.version)
import torch

# Thêm đường dẫn tới thư mục network
BASE_DIR = Path(__file__).resolve().parent
NETWORK_DIR = BASE_DIR / "network"
sys.path.append(str(NETWORK_DIR))

from utils import LoadModel, Text_to_vector, Predict, Output_to_text

# Tự động nạp danh sách model từ network/models (gồm 1 model ReLU và 5 model SiLU)
MODELS_DIR = NETWORK_DIR / "models"
AVAILABLE_MODELS = {
    p.stem: p
    for p in sorted(MODELS_DIR.glob("*/*.pth"))
}

@st.cache_resource
def get_model(model_name: str):
    path = AVAILABLE_MODELS[model_name]
    return LoadModel(str(path))

st.title("NPC")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Cài đặt")
    if st.button("Xóa lịch sử trò chuyện"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.bottom:
    col1, col2 = st.columns([4, 1], vertical_alignment="bottom")
    with col1:
        user_input = st.chat_input("Aa")
    with col2:
        selected_model = st.selectbox(
            label="Chọn model",
            options=list(AVAILABLE_MODELS.keys()),
            label_visibility="collapsed",
        )

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
   
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 6. Khu vực AI xử lý và trả về Output
    with st.chat_message("assistant"):
        with st.spinner("..."):
            model = get_model(selected_model)
            input_vector = Text_to_vector(user_input)
            output = Predict(model, input_vector)
            ai_response = Output_to_text(output)
            st.markdown(ai_response)
    
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
