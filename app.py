import streamlit as st
import time


st.title("NPC")

if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("Cài đặt")
    temp = st.slider("Độ sáng tạo (Temperature)", 0.0, 1.0, 0.7)
    if st.button("Xóa lịch sử trò chuyện"):
        st.session_state.messages = []
        st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if user_input := st.chat_input("Aa"):
    with st.chat_message("user"):
        st.markdown(user_input)
   
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 6. Khu vực AI xử lý và trả về Output
    with st.chat_message("assistant"):
        with st.spinner("..."):
            
            # ---> Sau này sẽ gọi model <---
            
            time.sleep(1)
            ai_response = f"Máy đã nhận được: {user_input}. Đây là phản hồi mẫu."
            st.markdown(ai_response)
    
   
    st.session_state.messages.append({"role": "assistant", "content": ai_response})