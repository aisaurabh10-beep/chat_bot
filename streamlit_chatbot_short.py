"""
BHARATHATECHNO AI - TOKEN WARNING VERSION
=========================================
1. Deep Search (30 chunks) for maximum detail.
2. "Detective Mode" to merge project details.
3. QUOTA TRAP: Instantly warns "Your token ended" if limits are hit.
"""

import streamlit as st
import os
import pickle
import warnings
import time
import logging
from typing import List, Dict

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['GRPC_VERBOSITY'] = 'ERROR'
logging.getLogger('google.generativeai').setLevel(logging.ERROR)

import google.generativeai as genai
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="BharathaTechno AI", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        font-size: 16px;
        line-height: 1.6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
        color: #0d47a1;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 5px solid #388e3c;
        color: #1b5e20;
    }
    .token-warning {
        background-color: #ffcdd2;
        color: #c62828;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #ef5350;
        font-weight: bold;
        text-align: center;
    }
    .debug-text {
        font-size: 12px;
        color: #666;
        font-family: monospace;
        background: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ddd;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# RESOURCES
# ---------------------------------------------------------
@st.cache_resource
def load_resources():
    try:
        db_path = "vector_db_faiss"
        index = faiss.read_index(os.path.join(db_path, "index.faiss"))
        with open(os.path.join(db_path, "chunks.pkl"), 'rb') as f:
            chunks = pickle.load(f)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return index, chunks, model
    except Exception as e:
        return None, None, None

# ---------------------------------------------------------
# LOGIC
# ---------------------------------------------------------
class RAGChatbot:
    def __init__(self):
        self.index, self.chunks, self.model = load_resources()

    def retrieve(self, query):
        if not self.model: return []
        try:
            vec = self.model.encode([query], normalize_embeddings=True)
            
            # Massive Context Window (30 Chunks)
            dists, idxs = self.index.search(vec, 30)
            
            results = []
            seen_content = set()
            
            for i in idxs[0]:
                if i < len(self.chunks):
                    txt = self.chunks[i]['text']
                    signature = txt[:100]
                    if signature not in seen_content:
                        results.append(self.chunks[i])
                        seen_content.add(signature)
            return results
        except: return []

    def generate(self, query, context, history):
        # Format Raw Context
        ctx_text = "\n\n".join([f"--- CHUNK ---\n{c['text']}" for c in context]) if context else "No context."
        
        hist_text = ""
        for h in history[-2:]:
            hist_text += f"User: {h['q']}\nSarah: {h['a']}\n"
        
        prompt = f"""
        You are Sarah, a Senior Consultant at BharathaTechno.
        
        RAW DATABASE CONTENT:
        {ctx_text}
        
        USER QUESTION: "{query}"
        
        INSTRUCTIONS:
        1. **GOAL:** The user wants a COMPLETE list of projects with ALL available details.
        2. **DETECTIVE WORK:** Look through ALL the chunks provided above. Information about a single project might be split across different chunks. Merge them.
        3. **DETAILS:** For each project, list the 'Technology Used', 'Features', and 'Outcome' if mentioned. 
        4. **NO LAZY FALLBACKS:** Do NOT say "Details coming soon" unless the text literally contains zero words about that project other than the title. Dig deeper!
        5. **FORMAT:** Use bold titles and bullet points.
        """

        models_to_try = [
        'gemini-2.5-flash-lite'
        ]

        last_error = ""

        for m in models_to_try:
            try:
                time.sleep(1)
                model = genai.GenerativeModel(m)
                
                # Max output tokens increased for long detailed lists
                response = model.generate_content(prompt, generation_config={"max_output_tokens": 2000})
                
                if response.text:
                    return response.text
                    
            except Exception as e:
                error_msg = str(e).lower()
                
                # --- NEW FEATURE: TOKEN WARNING CHECK ---
                # If error contains "429", "quota", or "limit", stop immediately.
                if "429" in error_msg or "quota" in error_msg or "limit" in error_msg:
                    return '<div class="token-warning">⚠️ Warning: Your token ended.<br><small>(You have reached the daily free limit for this model)</small></div>'
                
                last_error = str(e)
                continue

        return f"I apologize, connection error. ({last_error[:50]}...)"

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
def main():
    if 'messages' not in st.session_state: st.session_state.messages = []
    
    st.markdown('<h3 style="text-align:center;">🤖 BharathaTechno AI short</h3>', unsafe_allow_html=True)
    
    bot = RAGChatbot()

    with st.sidebar:
        api_key = st.text_input("Gemini API Key:", type="password")
        if api_key:
            genai.configure(api_key=api_key)
            st.success("✅ Connected")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    for msg in st.session_state.messages:
        role = "user-message" if msg['role'] == 'user' else "assistant-message"
        
        # Check if message is our special warning HTML
        if "token-warning" in msg['content']:
             st.markdown(msg['content'], unsafe_allow_html=True)
        else:
             st.markdown(f'<div class="chat-message {role}">{msg["content"]}</div>', unsafe_allow_html=True)
        
        if msg.get('raw_context'):
            with st.expander("🕵️ Debug: See Raw Database Content Used"):
                for i, chunk in enumerate(msg['raw_context']):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.markdown(f"<div class='debug-text'>{chunk['text'][:300]}...</div>", unsafe_allow_html=True)

    if user_input := st.chat_input("Ask Sarah..."):
        if not api_key:
            st.warning("Please enter API Key first.")
            return

        st.session_state.messages.append({'role': 'user', 'content': user_input})
        st.rerun()

    if st.session_state.messages and st.session_state.messages[-1]['role'] == 'user':
        with st.spinner("Sarah is gathering every detail..."):
            query = st.session_state.messages[-1]['content']
            context = bot.retrieve(query)
            
            history = [{'q': st.session_state.messages[i]['content'], 'a': st.session_state.messages[i+1]['content']} 
                       for i in range(0, len(st.session_state.messages)-1, 2)]
            
            response = bot.generate(query, context, history)
            
            st.session_state.messages.append({
                'role': 'assistant', 
                'content': response,
                'raw_context': context
            })
            st.rerun()

if __name__ == "__main__":
    main()