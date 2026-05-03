import streamlit as st
import time

st.set_page_config(page_title="AMD RAG")
st.title("🤖 AMD AI Assistant")

# 1. Import Backend (Now instant because we removed global code)
import backend_service

# 2. Initialize System (This is where we show the spinner)
@st.cache_resource
def start_system():
    # This calls the heavy function we created
    backend_service.initialize_rag_system()
    return True

# Show spinner while connecting/loading
with st.spinner("🚀 Connecting to Milvus & Loading Models..."):
    try:
        start_system()
        st.success("System Online")
    except Exception as e:
        st.error(f"Failed to start system: {e}")
        st.stop()

# 3. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about AMD..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        msg_placeholder.markdown("Thinking...")
        
        # Run Query
        response = backend_service.get_rag_response(prompt)
        answer = response.get("answer", "Error")
        
        msg_placeholder.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})