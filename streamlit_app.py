import streamlit as st
import time

# --- Page Config ---
st.set_page_config(
    page_title="AMD RAG Assistant",
    page_icon="🤖",
    layout="centered"
)

# --- Custom CSS Styling ---
st.markdown(
    """
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        /* Main app background */
        .stApp {
            background-color: #f8fafc;
            font-family: 'Inter', sans-serif;
        }
        
        /* Force all text to be dark */
        .stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp li {
            color: #1e293b !important;
        }
        
        /* Header styling */
        .header-container {
            text-align: center;
            padding: 1.5rem 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
        
        .header-title {
            color: #ffffff !important;
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .header-subtitle {
            color: #f1f5f9 !important;
            font-size: 1rem;
            margin-top: 0.5rem;
            font-weight: 400;
        }
        
        /* Success message */
        .stSuccess {
            background-color: #d1fae5 !important;
            border-left: 4px solid #10b981 !important;
            border-radius: 8px !important;
        }
        
        .stSuccess p, .stSuccess div {
            color: #065f46 !important;
        }
        
        /* Info message */
        .stInfo {
            background-color: #dbeafe !important;
            border-left: 4px solid #3b82f6 !important;
            border-radius: 8px !important;
        }
        
        .stInfo p, .stInfo div {
            color: #1e40af !important;
        }
        
        /* Error message */
        .stError {
            background-color: #fee2e2 !important;
            border-left: 4px solid #ef4444 !important;
            border-radius: 8px !important;
        }
        
        .stError p, .stError div {
            color: #991b1b !important;
        }
        
        /* Warning message */
        .stWarning {
            background-color: #fef3c7 !important;
            border-left: 4px solid #f59e0b !important;
            border-radius: 8px !important;
        }
        
        .stWarning p, .stWarning div {
            color: #92400e !important;
        }
        
        /* Chat messages container */
        [data-testid="stChatMessageContent"] {
            background-color: transparent !important;
        }
        
        [data-testid="stChatMessageContent"] p {
            color: #1e293b !important;
            line-height: 1.6;
        }
        
        /* User message styling */
        [data-testid="stChatMessage"]:has([data-testid="user-avatar"]) {
            background-color: #eff6ff !important;
            border-left: 3px solid #3b82f6 !important;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        /* Assistant message styling */
        [data-testid="stChatMessage"]:has([data-testid="assistant-avatar"]) {
            background-color: #ffffff !important;
            border-left: 3px solid #667eea !important;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        /* Caption text (latency) */
        .element-container:has(.stCaption) p {
            color: #64748b !important;
            font-size: 0.875rem;
            text-align: right;
        }
        
        /* Chat input */
        [data-testid="stChatInput"] input {
            color: #1e293b !important;
            background-color: #ffffff !important;
            border-radius: 20px !important;
        }
        
        [data-testid="stChatInput"] input::placeholder {
            color: #94a3b8 !important;
        }
        
        /* Divider */
        hr {
            margin: 1.5rem 0;
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #667eea, transparent);
            opacity: 0.3;
        }
        
        /* Spinner */
        .stSpinner > div {
            color: #667eea !important;
            text-align: center;
        }
        
        .stSpinner p {
            color: #667eea !important;
        }
        
        /* Animation */
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .stSuccess, .stInfo {
            animation: slideIn 0.5s ease-out;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Custom Header ---
st.markdown(
    """
    <div class="header-container">
        <h1 class="header-title">🤖 AMD AI Assistant</h1>
        <p class="header-subtitle">Powered by Advanced RAG Technology</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Lazy Load Backend with Caching ---
@st.cache_resource(show_spinner=False)
def load_rag_backend():
    """
    Imports the backend module.
    Since backend_service.py loads models at the top level,
    importing it triggers the model loading.
    We cache this so it only happens ONCE.
    """
    import backend_service
    return backend_service

# Load models with custom spinner
with st.spinner("🚀 Initializing AI Models... Please wait"):
    backend = load_rag_backend()

st.success("✅ System Ready! Start asking questions below.")
st.divider()

# --- Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show welcome message if no chat history
if len(st.session_state.messages) == 0:
    st.info("👋 **Welcome!** Ask me anything about AMD products, technologies, or documentation.")

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# --- Chat Input Area ---
if prompt := st.chat_input("💬 Ask me anything about AMD..."):
    # 1. Display User Message
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    
    # 2. Add to History
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Generate Response
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        
        # Show animated thinking message
        with message_placeholder.container():
            st.markdown("💭 _Analyzing your question..._")
        
        try:
            # Call the Backend Function via the cached module
            start_time = time.time()
            response_data = backend.get_rag_response(prompt)
            end_time = time.time()
            
            # Extract only the answer, ignore sources
            answer = response_data.get("answer", "I couldn't find an answer to that question.")
            
            # Display Answer
            message_placeholder.markdown(answer)
            
            # Show latency with emoji
            latency = end_time - start_time
            st.caption(f"⚡ Generated in **{latency:.2f}s**")

            # 4. Add Assistant Response to History (answer only, no sources)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            message_placeholder.error(f"❌ **Error:** {str(e)}")
            st.warning("Please try again or rephrase your question.")
