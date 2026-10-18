import streamlit as st


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
            label="",
            options=[""],
            label_visibility="collapsed",
            placeholder="",
        )

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
   
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 6. Khu vực AI xử lý và trả về Output
    with st.chat_message("assistant"):
        with st.spinner("..."):
            
            # ---> Sau này sẽ gọi model <---
            
            
            ai_response = f"Máy đã nhận được: {user_input}. Đây là phản hồi mẫu."
            st.markdown(ai_response)
    
   
    st.session_state.messages.append({"role": "assistant", "content": ai_response})