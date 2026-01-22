"""
PRODUCTION CHATBOT - BEAUTIFUL FORMATTED ANSWERS
================================================
Detailed natural answers with professional formatting
"""

import streamlit as st
import os
import json
import pickle
from datetime import datetime
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

st.set_page_config(
    page_title="BharathaTechno AI Assistant",
    page_icon="🤖",
    layout="wide"
)

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
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .assistant-message { 
        background-color: #ffffff;
    }
    .user-message strong {
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_faiss_index(db_path: str = "vector_db_faiss"):
    return faiss.read_index(os.path.join(db_path, 'index.faiss'))

@st.cache_resource
def load_chunks(db_path: str = "vector_db_faiss"):
    with open(os.path.join(db_path, 'chunks.pkl'), 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_embedding_model():
    import torch
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    model.eval()
    return model


class RAGChatbot:
    def __init__(self, db_path: str = "vector_db_faiss"):
        self.db_path = db_path
        self.index = None
        self.chunks = []
        self.embedding_model = None
        self.llm = None
        self.top_k = 10
        self.initialized = False
        self.model_name = None
    
    def initialize(self):
        try:
            self.index = load_faiss_index(self.db_path)
            self.chunks = load_chunks(self.db_path)
            self.embedding_model = load_embedding_model()
            self.initialized = True
            return True
        except Exception as e:
            st.error(f"Error: {e}")
            return False
    
    def setup_gemini(self, api_key: str):
        try:
            genai.configure(api_key=api_key)
            
            models_to_try = ['gemini-2.5-flash-lite']
            
            for model in models_to_try:
                try:
                    self.llm = genai.GenerativeModel(
                        model,
                        system_instruction="""You are a professional business consultant for BharathaTechno IT.

FORMATTING RULES:
1. Start with a brief intro paragraph (2-3 sentences)
2. Use clear section headers with ## for main points
3. Use bullet points (- or •) for lists of features/services
4. Use **bold** for important terms and service names
5. Include URLs as markdown links: [Service Name](url)
6. End with a helpful summary or call-to-action
7. Use emojis sparingly for visual interest (💡 🚀 ✨)

STRUCTURE TEMPLATE:
Brief introduction explaining the topic.

## Key Points
- Point 1 with **important details**
- Point 2 with [relevant link](url)

## Benefits/Features  
- Benefit 1
- Benefit 2

Final summary with next steps or contact info."""
                    )
                    self.model_name = model
                    st.success(f"✅ Using: {model}")
                    return True
                except:
                    continue
            
            return False
        except Exception as e:
            st.error(f"Error: {e}")
            return False
    
    def retrieve_context(self, query: str) -> List[Dict]:
        if not self.initialized:
            raise ValueError("Not initialized")
        
        try:
            query_embedding = self.embedding_model.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True
            )
            
            distances, indices = self.index.search(query_embedding, self.top_k * 2)
            
            seen_pages = set()
            results = []
            
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    page_id = chunk['metadata'].get('page_id')
                    
                    if page_id not in seen_pages:
                        seen_pages.add(page_id)
                        results.append({
                            'text': chunk['text'],
                            'metadata': chunk['metadata'],
                            'score': float(dist)
                        })
                    
                    if len(results) >= self.top_k:
                        break
            
            return results
        except Exception as e:
            st.error(f"Error: {e}")
            return []
    
    def generate_answer(self, query: str, context_chunks: List[Dict], conversation_history: List[Dict] = None) -> str:
        if not self.llm:
            raise ValueError("LLM not configured")
        
        try:
            # Build context
            context_parts = []
            urls = {}
            
            for i, chunk in enumerate(context_chunks, 1):
                meta = chunk['metadata']
                text = chunk['text']
                title = meta.get('title', 'Unknown')
                url = meta.get('url', 'N/A')
                
                context_parts.append(f"SOURCE {i}: {title}\nURL: {url}\nCONTENT: {text}\n{'-'*80}")
                
                if url and url != 'N/A':
                    urls[title] = url
            
            full_context = "\n\n".join(context_parts)
            
            # Conversation history
            history_text = ""
            if conversation_history:
                recent = conversation_history[-2:]
                hist_parts = []
                for h in recent:
                    hist_parts.append(f"User asked: \"{h['question']}\"\nYou answered: \"{h['answer'][:150]}...\"")
                history_text = "\n\n".join(hist_parts)
            
            # FORMATTED PROMPT
            prompt = f"""Answer this question with BEAUTIFUL FORMATTING using markdown.

CRITICAL FORMATTING REQUIREMENTS:
1. Start with a friendly 2-sentence introduction
2. Use ## headers for main sections (like "## Our Services", "## Key Features")
3. Use bullet points (-) for lists
4. Use **bold** for service names and key terms
5. Include URLs as [Service Name](url) links
6. Add appropriate emojis (💡 🚀 ✨ ⚙️) for visual interest
7. End with a helpful summary

EXAMPLE FORMAT:
Brief intro explaining the topic clearly and professionally.

## Main Section Header
- **Key Point 1**: Explanation with details
- **Key Point 2**: More information [with link](url)

## Benefits
- Benefit 1 described clearly
- Benefit 2 with specifics

💡 *Summary or call-to-action here*

CONTEXT:
{full_context}

AVAILABLE LINKS:
{chr(10).join([f"- {title}: {url}" for title, url in urls.items()])}

{"CONVERSATION:" + chr(10) + history_text if history_text else ""}

USER QUESTION: {query}

Now write a well-formatted, professional answer:"""

            config = genai.types.GenerationConfig(
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                max_output_tokens=800,
            )
            
            # Try to generate response with immediate error detection
            try:
                response = self.llm.generate_content(
                    prompt, 
                    generation_config=config, 
                    request_options={'timeout': 60}
                )
                
                answer = ""
                if hasattr(response, 'text') and response.text:
                    answer = response.text.strip()
                elif hasattr(response, 'candidates') and response.candidates:
                    parts = response.candidates[0].content.parts
                    if parts:
                        answer = parts[0].text.strip()
                
                if answer:
                    # Ensure it has some formatting
                    if '##' not in answer and len(answer) > 300:
                        answer = self.add_basic_formatting(answer, context_chunks)
                    return answer
                    
            except Exception as api_error:
                # Immediately check for quota/rate limit errors
                error_str = str(api_error).lower()
                
                # QUOTA EXCEEDED - Most common
                if any(word in error_str for word in ['429', 'quota', 'resource exhausted', 'rate limit', 'limit exceeded']):
                    return """## ⚠️ Token Limit Reached!

**Your Gemini API quota has been exceeded.**

### 🔴 What Happened:
You've hit the **free tier limit** for Gemini API:
- **15 requests per minute**, OR
- **1,500 requests per day**

### ✅ Quick Solutions:

**Option 1: Wait & Retry** ⏰
- Rate limit resets in **60 seconds**
- Daily quota resets at **midnight UTC**
- Just wait and try again!

**Option 2: Upgrade to Paid** 💳
- Get 1,000+ requests/minute
- Only **~$0.01 per 1,000 requests**
- Upgrade at: [Google AI Studio](https://makersuite.google.com/app/apikey)

**Option 3: New Free Key** 🔑
- Use a different Google account
- Get another free API key
- Click **Disconnect** and enter new key

### 💡 For Your Team:
With 5-20 users, upgrading costs **$3-5/month total**. Worth it for smooth operation!

---
*Free tier is great for testing, but production needs paid tier.*"""

                # INVALID API KEY
                elif any(word in error_str for word in ['api key', 'invalid', 'authentication', '401', '403', 'permission']):
                    return """## ❌ API Key Invalid

**Your API key is not working.**

### Possible Reasons:
- Key expired (unused for 90+ days)
- Key was deleted
- Wrong key entered
- Permissions issue

### Fix It:
1. **Disconnect** (sidebar button)
2. **Get new key**: [Google AI Studio](https://makersuite.google.com/app/apikey)
3. **Reconnect** with new key

💡 Make sure to copy the entire key!"""

                # TIMEOUT
                elif 'timeout' in error_str:
                    return """## ⏱️ Request Timeout

**The request took too long.**

### Try:
- Ask a simpler/shorter question
- Rephrase your question
- Wait a moment and try again

💡 Gemini API might be experiencing high load."""

                # CONTENT BLOCKED
                elif any(word in error_str for word in ['blocked', 'safety', 'filtered']):
                    return """## 🚫 Content Blocked

**The response was blocked by safety filters.**

### What to do:
- Try rephrasing your question
- Ask about a different topic
- This is a Gemini API safety feature

💡 Your quota is fine - just the content was flagged."""

                # UNKNOWN ERROR
                else:
                    return f"""## ❌ API Error

**Something went wrong with the Gemini API.**

### Error Details:
```
{str(api_error)[:300]}
```

### Try:
- Refresh the page
- Try again in a moment
- Check [Gemini Status](https://status.cloud.google.com/)

💡 If this persists, your API key might need to be regenerated."""
            
            # If we get here, return fallback
            return self.create_formatted_fallback(context_chunks, query)
            
        except Exception as e:
            # Outer exception handler for any other errors
            print(f"Outer error: {e}")
    
    def add_basic_formatting(self, text: str, context_chunks: List[Dict]) -> str:
        """Add basic formatting to plain text"""
        lines = text.split('. ')
        
        if len(lines) > 4:
            # Add a header after intro
            formatted = f"{lines[0]}. {lines[1]}.\n\n## Key Information\n\n"
            formatted += '. '.join(lines[2:])
        else:
            formatted = text
        
        # Add link at the end if not present
        if 'http' not in formatted and context_chunks:
            url = context_chunks[0]['metadata'].get('url', '')
            title = context_chunks[0]['metadata'].get('title', '')
            if url and url != 'N/A':
                formatted += f"\n\n📚 **Learn More**: [{title}]({url})"
        
        return formatted
    
    def create_formatted_fallback(self, context_chunks: List[Dict], query: str) -> str:
        """Create a beautifully formatted fallback"""
        
        if not context_chunks:
            return f"""I'd be happy to help you learn more about {query}!

## Get in Touch
For detailed information, please visit our website or contact us directly:

- 🌐 **Website**: [BharathaTechno](https://bharathatechno.com)
- 📧 **Contact**: [Get in Touch](https://bharathatechno.com/contact)

💡 *Feel free to ask me another question!*"""
        
        top_chunk = context_chunks[0]
        title = top_chunk['metadata'].get('title', '')
        url = top_chunk['metadata'].get('url', '')
        content = top_chunk['text'][:400]
        
        response = f"""Let me share what we offer regarding **{query}**.

## Overview
{content}

## Related Services"""
        
        if len(context_chunks) > 1:
            for chunk in context_chunks[1:4]:
                t = chunk['metadata'].get('title', '')
                u = chunk['metadata'].get('url', '')
                if u and u != 'N/A':
                    response += f"\n- [{t}]({u})"
        
        if url and url != 'N/A':
            response += f"\n\n📚 **Learn More**: [{title}]({url})"
        
        response += "\n\n💡 *Ask me anything else about our services!*"
        
        return response


def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chatbot_ready' not in st.session_state:
        st.session_state.chatbot_ready = False
    if 'api_key_set' not in st.session_state:
        st.session_state.api_key_set = False


def setup_chatbot(api_key: str):
    try:
        chatbot = RAGChatbot()
        with st.spinner("Loading..."):
            if not chatbot.initialize() or not chatbot.setup_gemini(api_key):
                return False
        
        st.session_state.chatbot = chatbot
        st.session_state.chatbot_ready = True
        st.session_state.api_key_set = True
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False


def display_message(message: Dict):
    role = message['role']
    content = message['content']
    is_error = message.get('is_error', False)
    
    if role == 'user':
        st.markdown(f'<div class="chat-message user-message"><strong>🙋 You:</strong><br>{content}</div>', unsafe_allow_html=True)
    else:
        # Add warning styling for error messages
        error_class = ' style="border-left: 4px solid #ff9800;"' if is_error else ''
        st.markdown(f'<div class="chat-message assistant-message"{error_class}><strong>🤖 Assistant:</strong>', unsafe_allow_html=True)
        
        # Render markdown content
        st.markdown(content)
        
        if 'sources' in message and not is_error:
            with st.expander("📚 View Sources"):
                for i, src in enumerate(message['sources'][:5], 1):
                    st.markdown(f"**{i}. {src['title']}**")
                    st.markdown(f"🔗 [{src['url']}]({src['url']})")
                    st.markdown("---")
        
        st.markdown('</div>', unsafe_allow_html=True)


def main():
    initialize_session_state()
    
    st.markdown('<div class="main-header">🤖 BharathaTechno AI Assistant</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        if not st.session_state.api_key_set:
            st.info("🔑 Get key: https://makersuite.google.com/app/apikey")
            api_key = st.text_input("Gemini API Key:", type="password")
            
            if st.button("🚀 Connect", type="primary"):
                if api_key and setup_chatbot(api_key):
                    st.rerun()
        else:
            st.success(f"✅ Connected")
            if st.session_state.chatbot:
                st.info(f"🤖 Model: {st.session_state.chatbot.model_name}")
            if st.button("🔌 Disconnect"):
                st.session_state.api_key_set = False
                st.session_state.chatbot_ready = False
                st.session_state.chatbot = None
                st.rerun()
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("💾 Export"):
                st.info("Coming soon!")
        
        st.markdown("---")
        st.metric("💬 Messages", len(st.session_state.messages))
        
        st.markdown("---")
        st.markdown("### 💡 Quick Questions")
        samples = [
            "What services do you offer?",
            "Tell me about staff augmentation",
            "Help me build a website",
            "MERN stack development",
            "Cloud services"
        ]
        
        for q in samples:
            if st.button(f"💭 {q}", key=q, use_container_width=True):
                st.session_state.sample_q = q
    
    if not st.session_state.chatbot_ready:
        st.info("👈 Enter your API key to start chatting")
        
        st.markdown("""
        ## 👋 Welcome to BharathaTechno AI Assistant!
        
        I can help you with:
        
        - 🔍 **Services Information** - Learn about our offerings
        - 💼 **Project Details** - Understand our capabilities
        - 🛠️ **Technology Stack** - Explore our tech expertise
        - 📞 **Contact Info** - Get in touch with us
        
        ### 🚀 Getting Started
        1. Get a FREE API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Enter it in the sidebar
        3. Start asking questions!
        """)
    else:
        for msg in st.session_state.messages:
            display_message(msg)
        
        user_input = st.session_state.pop('sample_q', None)
        user_question = st.chat_input("💬 Ask me anything about BharathaTechno...")
        
        if user_question or user_input:
            question = user_question or user_input
            st.session_state.messages.append({'role': 'user', 'content': question})
            
            with st.spinner("🤔 Thinking..."):
                try:
                    context = st.session_state.chatbot.retrieve_context(question)
                    
                    conv_hist = []
                    for i in range(0, len(st.session_state.messages) - 1, 2):
                        if i + 1 < len(st.session_state.messages):
                            conv_hist.append({
                                'question': st.session_state.messages[i]['content'],
                                'answer': st.session_state.messages[i + 1]['content']
                            })
                    
                    answer = st.session_state.chatbot.generate_answer(question, context, conv_hist)
                    
                    # Check if answer is an error message
                    is_error = answer.startswith("## ⚠️") or answer.startswith("## ❌") or answer.startswith("## ⏱️")
                    
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': answer,
                        'sources': [{'title': c['metadata'].get('title'), 'url': c['metadata'].get('url')} for c in context[:5]],
                        'is_error': is_error
                    })
                    
                    # Show notification for errors
                    if is_error:
                        if "Quota Exceeded" in answer:
                            st.error("⚠️ API Quota Exceeded - See message below for solutions")
                        elif "API Key Error" in answer:
                            st.error("❌ API Key Invalid - Please reconnect with a new key")
                        elif "Timeout" in answer:
                            st.warning("⏱️ Request Timeout - Try a simpler question")
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")


if __name__ == "__main__":
    main()