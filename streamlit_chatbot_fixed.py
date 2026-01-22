"""
STREAMLIT CHATBOT - FIXED FOR MULTI-USER/NETWORK ACCESS
========================================================
Fixed issues:
- Proper session state management
- Embedding model loaded once per session
- Better error handling for network access
- Thread-safe operations
"""

import streamlit as st
import os
import json
import pickle
from datetime import datetime
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Vector search
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Google Gemini LLM
import google.generativeai as genai

# Page config
st.set_page_config(
    page_title="BharathaTechno AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .source-box {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# GLOBAL CACHED RESOURCES (shared across all users)
@st.cache_resource
def load_faiss_index(db_path: str = "vector_db_faiss"):
    """Load FAISS index - cached globally"""
    index_path = os.path.join(db_path, 'index.faiss')
    return faiss.read_index(index_path)


@st.cache_resource
def load_chunks(db_path: str = "vector_db_faiss"):
    """Load chunks metadata - cached globally"""
    chunks_path = os.path.join(db_path, 'chunks.pkl')
    with open(chunks_path, 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_embedding_model():
    """Load embedding model - cached globally"""
    import torch
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    model.eval()
    return model


class RAGChatbot:
    def __init__(self, db_path: str = "vector_db_faiss"):
        """Initialize the RAG chatbot"""
        self.db_path = db_path
        self.index = None
        self.chunks = []
        self.embedding_model = None
        self.llm = None
        self.top_k = 5
        self.initialized = False
    
    def initialize(self):
        """Initialize all components"""
        try:
            # Load shared resources (cached globally)
            self.index = load_faiss_index(self.db_path)
            self.chunks = load_chunks(self.db_path)
            self.embedding_model = load_embedding_model()
            self.initialized = True
            return True
        except Exception as e:
            st.error(f"Error initializing chatbot: {e}")
            return False
    
    def setup_gemini(self, api_key: str):
        """Setup Google Gemini API"""
        try:
            genai.configure(api_key=api_key)
            self.llm = genai.GenerativeModel('gemini-flash-latest')
            return True
        except Exception as e:
            st.error(f"Error setting up Gemini: {e}")
            return False
    
    def retrieve_context(self, query: str) -> List[Dict]:
        """Retrieve relevant chunks for a query"""
        if not self.initialized or self.embedding_model is None:
            raise ValueError("Chatbot not initialized. Call initialize() first.")
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Search in FAISS
            distances, indices = self.index.search(query_embedding, self.top_k)
            
            # Get chunks
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    results.append({
                        'text': chunk['text'],
                        'metadata': chunk['metadata'],
                        'score': float(dist)
                    })
            
            return results
        except Exception as e:
            st.error(f"Error retrieving context: {e}")
            return []
    
    def generate_answer(self, query: str, context_chunks: List[Dict], conversation_history: List[Dict] = None) -> str:
        """Generate answer using Gemini with retrieved context"""
        
        if not self.llm:
            raise ValueError("Gemini not configured. Call setup_gemini() first.")
        
        try:
            # Build context from chunks
            context_parts = []
            for i, chunk in enumerate(context_chunks, 1):
                metadata = chunk['metadata']
                text = chunk['text']
                
                context_parts.append(
                    f"[Source {i}] {metadata.get('title', 'Unknown')}\n"
                    f"URL: {metadata.get('url', 'N/A')}\n"
                    f"Content: {text}\n"
                )
            
            context_text = "\n---\n".join(context_parts)
            
            # Build conversation history context
            history_text = ""
            if conversation_history:
                history_parts = []
                for msg in conversation_history[-3:]:  # Last 3 exchanges
                    history_parts.append(f"User: {msg['question']}\nAssistant: {msg['answer']}")
                history_text = "\n\n".join(history_parts)
            
            # Create prompt
            prompt = f"""You are a helpful AI assistant for BharathaTechno IT company.
Answer the user's question based on the provided context from the company website.

IMPORTANT RULES:
1. Only use information from the context provided below
2. If the context doesn't contain enough information, say "I don't have enough information about that. Please contact us at https://bharathatechno.com/contact"
3. Be specific and helpful
4. Keep answers concise but complete (2-4 sentences)
5. If relevant, mention the URL where users can find more details
6. Use a friendly, professional tone

CONTEXT FROM WEBSITE:
{context_text}

{"RECENT CONVERSATION:" if history_text else ""}
{history_text}

CURRENT QUESTION: {query}

ANSWER:"""

            response = self.llm.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Error generating answer: {e}"


def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    
    if 'chatbot_ready' not in st.session_state:
        st.session_state.chatbot_ready = False
    
    if 'api_key_set' not in st.session_state:
        st.session_state.api_key_set = False
    
    if 'initialization_done' not in st.session_state:
        st.session_state.initialization_done = False


def setup_chatbot(api_key: str):
    """Setup the chatbot with API key"""
    try:
        # Create chatbot instance
        chatbot = RAGChatbot()
        
        # Initialize (loads vector DB and models)
        with st.spinner("🔄 Loading vector database..."):
            if not chatbot.initialize():
                return False
        
        # Setup Gemini
        with st.spinner("🔄 Connecting to Gemini AI..."):
            if not chatbot.setup_gemini(api_key):
                return False
        
        st.session_state.chatbot = chatbot
        st.session_state.chatbot_ready = True
        st.session_state.api_key_set = True
        st.session_state.initialization_done = True
        
        return True
    except Exception as e:
        st.error(f"❌ Error setting up chatbot: {e}")
        st.error("Make sure vector_db_faiss/ folder exists in the same directory")
        return False


def display_message(message: Dict):
    """Display a chat message"""
    role = message['role']
    content = message['content']
    
    if role == 'user':
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🙋 You:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 Assistant:</strong><br>
            {content}
        """, unsafe_allow_html=True)
        
        # Display sources if available
        if 'sources' in message:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message['sources'][:3], 1):
                    st.markdown(f"""
                    **{i}. {source['title']}**  
                    🔗 [{source['url']}]({source['url']})  
                    Relevance: {source['score']:.2%}
                    """)
        
        st.markdown("</div>", unsafe_allow_html=True)


def export_conversation():
    """Export conversation history"""
    if st.session_state.messages:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"chat_history_{timestamp}.txt"
        
        content = f"BharathaTechno AI Assistant - Chat History\n"
        content += f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += "=" * 60 + "\n\n"
        
        for msg in st.session_state.messages:
            if msg['role'] == 'user':
                content += f"You: {msg['content']}\n\n"
            else:
                content += f"Assistant: {msg['content']}\n"
                if 'sources' in msg:
                    content += "\nSources:\n"
                    for i, source in enumerate(msg['sources'][:3], 1):
                        content += f"{i}. {source['title']}\n   {source['url']}\n"
                content += "\n" + "-" * 60 + "\n\n"
        
        return content, filename
    return None, None


def main():
    """Main Streamlit app"""
    
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🤖 BharathaTechno AI Assistant</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Key input
        if not st.session_state.api_key_set:
            st.markdown("### 🔑 Setup")
            st.info("Get your FREE API key at: https://makersuite.google.com/app/apikey")
            
            api_key = st.text_input(
                "Enter Gemini API Key:",
                type="password",
                help="Your API key will not be stored"
            )
            
            if st.button("Connect", type="primary"):
                if api_key:
                    if setup_chatbot(api_key):
                        st.success("✅ Chatbot ready!")
                        st.rerun()
                else:
                    st.error("Please enter an API key")
        else:
            st.success("✅ Connected")
            
            if st.button("Disconnect"):
                st.session_state.api_key_set = False
                st.session_state.chatbot_ready = False
                st.session_state.chatbot = None
                st.session_state.initialization_done = False
                st.rerun()
        
        st.markdown("---")
        
        # Chat controls
        st.markdown("### 💬 Chat Controls")
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("📥 Export Chat"):
            content, filename = export_conversation()
            if content:
                st.download_button(
                    label="Download Chat History",
                    data=content,
                    file_name=filename,
                    mime="text/plain"
                )
            else:
                st.info("No messages to export")
        
        st.markdown("---")
        
        # Statistics
        st.markdown("### 📊 Statistics")
        st.metric("Total Messages", len(st.session_state.messages))
        st.metric("Your Questions", len([m for m in st.session_state.messages if m['role'] == 'user']))
        
        st.markdown("---")
        
        # Sample questions
        st.markdown("### 💡 Try Asking:")
        sample_questions = [
            "What services do you offer?",
            "Tell me about MERN stack development",
            "How can I contact you?",
            "What is B-Store?",
            "Do you provide cloud services?"
        ]
        
        for q in sample_questions:
            if st.button(f"💭 {q}", key=f"sample_{q}"):
                st.session_state.sample_question = q
    
    # Main chat area
    if not st.session_state.chatbot_ready:
        st.info("👈 Please enter your Gemini API key in the sidebar to start chatting")
        
        # Show welcome message
        st.markdown("""
        ## Welcome! 👋
        
        I'm your AI assistant for BharathaTechno IT. I can help you with:
        
        - 🔍 Information about our services
        - 💼 Details about our projects
        - 🛠️ Technology stack and capabilities
        - 📞 Contact information
        - 📦 Product information
        
        **To get started:**
        1. Get a FREE API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Enter it in the sidebar
        3. Start asking questions!
        """)
    else:
        # Display chat messages
        for message in st.session_state.messages:
            display_message(message)
        
        # Handle sample question
        if 'sample_question' in st.session_state:
            user_input = st.session_state.sample_question
            del st.session_state.sample_question
        else:
            user_input = None
        
        # Chat input
        user_question = st.chat_input("Ask me anything about BharathaTechno...")
        
        if user_question or user_input:
            question = user_question if user_question else user_input
            
            # Add user message
            st.session_state.messages.append({
                'role': 'user',
                'content': question
            })
            
            # Generate response
            with st.spinner("🤔 Thinking..."):
                try:
                    # Check if chatbot is properly initialized
                    if not st.session_state.chatbot or not st.session_state.chatbot.initialized:
                        st.error("Chatbot not initialized properly. Please refresh and reconnect.")
                        st.stop()
                    
                    # Retrieve context
                    context_chunks = st.session_state.chatbot.retrieve_context(question)
                    
                    if not context_chunks:
                        st.error("Could not retrieve relevant information. Please try again.")
                        st.stop()
                    
                    # Generate answer
                    conversation_history = [
                        {'question': m['content'], 'answer': st.session_state.messages[i+1]['content']}
                        for i, m in enumerate(st.session_state.messages[:-1])
                        if m['role'] == 'user' and i+1 < len(st.session_state.messages)
                    ]
                    
                    answer = st.session_state.chatbot.generate_answer(
                        question,
                        context_chunks,
                        conversation_history
                    )
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': answer,
                        'sources': [
                            {
                                'title': chunk['metadata'].get('title', 'Unknown'),
                                'url': chunk['metadata'].get('url', 'N/A'),
                                'score': chunk['score']
                            }
                            for chunk in context_chunks[:3]
                        ]
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.error("Please refresh the page and try again.")


if __name__ == "__main__":
    main()