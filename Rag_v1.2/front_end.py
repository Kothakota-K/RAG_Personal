import streamlit as st
import pandas as pd
from backend_service import retrieve_chunks

st.set_page_config(page_title="Milvus Knowledge Retriever", layout="wide")

# --- Initialize Session State for History ---
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# --- CSS for better visuals ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .chunk-card {
        background-color: #ffffff;
        color: #333333; /* FIXED: Forces text to be dark grey/black */
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .chunk-card h4 {
        color: #000000; /* Forces headers to be pure black */
        margin-top: 0;
    }
    .chunk-card p {
        color: #333333; /* Ensures paragraph text is dark */
        font-size: 16px;
        line-height: 1.6;
    }
    .source-link {
        font-size: 0.85em;
        color: #555;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Search History ---
with st.sidebar:
    st.header("🕒 Search History")
    if st.session_state.search_history:
        # Show latest searches at the top
        for i, prev_query in enumerate(reversed(st.session_state.search_history)):
            st.text(f"{i+1}. {prev_query}")
        
        if st.button("Clear History"):
            st.session_state.search_history = []
            st.rerun()
    else:
        st.info("No recent searches.")

# --- Header ---
st.title("🔍 Milvus Knowledge Retriever")
st.markdown("Search your vector database directly without LLM summarization.")
st.divider()

# --- Search Section ---
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input("Enter your search query:", placeholder="e.g., What are the latest AMD GPU specs?")

with col2:
    st.write("") # Spacer
    st.write("") # Spacer
    search_button = st.button("Search Knowledge Base", type="primary", use_container_width=True)

# --- Results Section ---
if search_button and query:
    # Save query to history if it's not the same as the last one
    if not st.session_state.search_history or st.session_state.search_history[-1] != query:
        st.session_state.search_history.append(query)

    with st.spinner("Searching Milvus Vector Database..."):
        try:
            # Call the backend function
            results = retrieve_chunks(query)
            
            if not results:
                st.warning("No relevant documents found for this query.")
            else:
                st.success(f"Found {len(results)} relevant chunks.")
                
                # Display results
                for item in results:
                    with st.container():
                        st.markdown(f"""
                        <div class="chunk-card">
                            <h4>Chunk #{item['chunk_id']}</h4>
                            <p>{item['content']}</p>
                            <hr>
                            <p class="source-link">🔗 Source: <a href="{item['source_url']}" target="_blank">{item['source_url']}</a></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
        except Exception as e:
            st.error(f"An error occurred: {e}")

elif search_button and not query:
    st.warning("Please enter a query first.")