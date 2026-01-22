"""
BHARATHATECHNO AI ASSISTANT - OPTIMIZED VERSION
================================================
Combining natural conversation flow with accurate, crisp information
- Context-aware responses (doesn't repeat greetings)
- Links included where relevant
- Crisp, not too long, not too short
- Positive, helpful tone
- Only shares information from the website
"""

import streamlit as st
import os
import pickle
import warnings
import time
import logging
from typing import List, Dict
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['GRPC_VERBOSITY'] = 'ERROR'
logging.getLogger('google.generativeai').setLevel(logging.ERROR)

import google.generativeai as genai
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="BharathaTechno AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .chat-message {
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-left: 4px solid #5568d3;
    }
    .assistant-message {
        background-color: #f8f9fa;
        border-left: 4px solid #388e3c;
        color: #212529;
    }
    .user-message strong {
        color: #fff;
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
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# RESOURCE LOADING
# ---------------------------------------------------------
@st.cache_resource
def load_resources():
    """Load FAISS index, chunks, and embedding model"""
    try:
        db_path = "vector_db_faiss"
        index = faiss.read_index(os.path.join(db_path, "index.faiss"))
        with open(os.path.join(db_path, "chunks.pkl"), 'rb') as f:
            chunks = pickle.load(f)
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        model.eval()
        return index, chunks, model
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None

# ---------------------------------------------------------
# CHATBOT CLASS
# ---------------------------------------------------------
class RAGChatbot:
    def __init__(self):
        self.index, self.chunks, self.model = load_resources()
        self.initialized = self.index is not None

    def retrieve_context(self, query: str, top_k: int = 18) -> List[Dict]:
        """Retrieve relevant context chunks with smart deduplication"""
        if not self.model:
            return []
        
        try:
            vec = self.model.encode([query], normalize_embeddings=True)
            dists, idxs = self.index.search(vec, top_k * 2)
            
            results = []
            seen_content = set()
            
            for idx in idxs[0]:
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    txt = chunk['text']
                    # Use longer signature (200 chars) to allow different content from same page
                    signature = txt[:200]
                    
                    # Only deduplicate if content is truly similar
                    if signature not in seen_content:
                        results.append(chunk)
                        seen_content.add(signature)
                    
                    if len(results) >= top_k:
                        break
            
            return results
        except Exception as e:
            st.error(f"Retrieval error: {e}")
            return []

    def _validate_completeness(self, query: str, answer: str, context: List[Dict]) -> str:
        """
        Validate if response is complete - especially for office locations and project links
        If incomplete, add a warning or correction
        """
        query_lower = query.lower()
        
        # Check if this is an office/location question
        office_keywords = ['office', 'location', 'address', 'where', 'contact']
        is_office_question = any(keyword in query_lower for keyword in office_keywords)
        
        if is_office_question:
            # Check if both India and Germany are mentioned in the answer
            has_india = 'india' in answer.lower() or 'pune' in answer.lower()
            has_germany = 'germany' in answer.lower() or 'fellbach' in answer.lower()
            
            # Check if both exist in context
            context_text = ' '.join([c['text'].lower() for c in context])
            india_in_context = 'india' in context_text or 'pune' in context_text
            germany_in_context = 'germany' in context_text or 'fellbach' in context_text
            
            # If both offices in context but answer only mentions one, fix it
            if india_in_context and germany_in_context:
                if has_india and not has_germany:
                    # Missing Germany office - add it
                    answer += "\n\n**Germany Office:**\nBruckwiesenweg 2, 70734 Fellbach, Germany\n📧 kourosh.bagherian@bharathatechno.de"
                elif has_germany and not has_india:
                    # Missing India office - add it
                    answer = f"**India Office:**\nSunshree Woods, Office F28-32, NIBM Rd, Kondhwa, Pune, Maharashtra 411048\n📧 info@bharathatechno.com\n📞 +91 932 563 6885\n\n" + answer
                elif not has_india and not has_germany:
                    # Missing both! Replace with complete answer
                    answer = """We have offices in two locations:

**India Office:**
Sunshree Woods, Office F28-32, NIBM Rd, Kondhwa, Pune, Maharashtra 411048
📧 info@bharathatechno.com
📞 +91 932 563 6885

**Germany Office:**
Bruckwiesenweg 2, 70734 Fellbach, Germany
📧 kourosh.bagherian@bharathatechno.de

Feel free to reach out to either location!"""
        
        # Check if this is a projects/portfolio question
        project_keywords = ['project', 'portfolio', 'work', 'completed', 'case study']
        is_project_question = any(keyword in query_lower for keyword in project_keywords)
        
        if is_project_question:
            # List of known projects that should have links
            projects_with_links = {
                'trendy threads': 'https://bharathatechno.com/work/trendy-threads',
                'nikita': 'https://bharathatechno.com/work/nikitas-curry-corner',
                'urban': 'https://bharathatechno.com/work/urban-axis',
                'sachniti': 'https://bharathatechno.com/work/sachniti',
                'train': 'https://bharathatechno.com/work/train-2-gain',
                'everwin': 'https://bharathatechno.com/work/everwinpt',
                'snumero': 'https://bharathatechno.com/work/snumero',
                'r3': 'https://bharathatechno.com/work/r3constructor',
                'intignus': 'https://bharathatechno.com/work/intignus-biotech',
            }
            
            # Check if answer mentions projects without links
            answer_lower = answer.lower()
            for project_name, project_url in projects_with_links.items():
                # If project is mentioned but link is not
                if project_name in answer_lower and project_url not in answer:
                    # This is a soft hint - don't force it, as it might break formatting
                    # The prompt should handle this, but we log it for debugging
                    pass
        
        return answer

    def generate_response(self, query: str, context: List[Dict], history: List[Dict]) -> str:
        """Generate AI response with context awareness"""
        
        # Build context
        context_text = "\n\n".join([
            f"SOURCE: {c['metadata'].get('title', 'N/A')}\nURL: {c['metadata'].get('url', 'N/A')}\nCONTENT: {c['text']}"
            for c in context
        ]) if context else "No relevant information found."
        
        # Collect available URLs
        urls = {}
        for c in context:
            url = c['metadata'].get('url', '')
            title = c['metadata'].get('title', '')
            if url and url != 'N/A' and title:
                urls[title] = url
        
        # Format conversation history
        history_text = ""
        if len(history) > 0:
            recent_history = history[-3:]  # Last 3 exchanges
            history_text = "\nRECENT CONVERSATION:\n"
            for h in recent_history:
                history_text += f"User: {h['q']}\nAssistant: {h['a'][:100]}...\n\n"
        
        # Check if this is first message
        is_first_message = len(history) == 0
        
        # Create dynamic prompt with强化 completeness protocol
        prompt = f"""You are a helpful AI assistant for BharathaTechno IT.

CRITICAL RULES:
1. ONLY use information from the provided CONTEXT below - never invent or add information not in the database
2. If context doesn't contain the answer, politely say you don't have that specific information and suggest they contact the team
3. Be conversational and natural - if this is a follow-up question, DON'T repeat greetings or introductions
4. Keep responses CRISP - aim for 150-250 words unless the question requires more detail
5. Include relevant links using markdown format: [Link Text](url)
6. Use bullet points for lists, but avoid excessive formatting
7. Be positive and helpful in tone
8. If asked about AI services or projects not in context, clearly state BharathaTechno doesn't currently offer that

🚨 COMPLETENESS PROTOCOL - MANDATORY (READ CAREFULLY):

**OFFICE LOCATIONS - CRITICAL RULE:**
BharathaTechno has TWO offices: India AND Germany.
When ANYONE asks "where is office" or "where is your office" or "office location":
→ YOU MUST SHOW BOTH OFFICES
→ NEVER show just one office
→ Even if question is singular ("office") not plural ("offices")

**CORRECT FORMAT for office questions:**
"We have offices in two locations:

**India Office:**
Sunshree Woods, Office F28-32, NIBM Rd, Kondhwa, Pune, Maharashtra 411048
📧 info@bharathatechno.com
📞 +91 932 563 6885

**Germany Office:**
Bruckwiesenweg 2, 70734 Fellbach, Germany
📧 kourosh.bagherian@bharathatechno.de"

**WRONG - NEVER DO THIS:**
"We have an office in India: [address]" ❌ INCOMPLETE!
"Our office is located in Pune" ❌ MISSING GERMANY!

**PROJECTS/PORTFOLIO - CRITICAL RULE:**
When listing projects, ALWAYS include the project link if available in AVAILABLE LINKS section.

**CORRECT FORMAT for projects:**
"* **Trendy Threads**: E-commerce web app - [View Project](url)
* **Nikita's Curry Corner**: Cloud kitchen platform - [View Project](url)"

**WRONG - NEVER DO THIS:**
"* Trendy Threads: E-commerce web app" ❌ MISSING LINK!
"* We developed Sachniti" ❌ NO LINK PROVIDED!

→ Check AVAILABLE LINKS section below for ALL project URLs
→ If project URL exists, include it as [View Project](url) or [Project Name](url)
→ Don't just mention project name without link if link is available

**GENERAL COMPLETENESS RULES:**
1. **Scan context for ALL instances** - If multiple items exist, LIST ALL:
   - Office locations → Show BOTH India AND Germany (mandatory)
   - Services → Mention ALL key services (5+ services)
   - Projects → Multiple project examples (3+ projects) WITH LINKS
   - Contact methods → ALL ways to reach
   
2. **Even Singular Questions Need Complete Answers:**
   - "Where is office?" → Show BOTH offices
   - "What service?" → List ALL services
   - "A project?" → Mention multiple projects WITH LINKS
   
3. **Presentation Style:**
   - Clear structure with bullet points
   - Equal prominence for all items
   - Don't pick favorites
   - Include links for projects/services when available
   
4. **Only Exception:** If user explicitly says:
   - "Give me ONE example"
   - "Just show ONE"
   - "Pick ONE for me"

⚠️ MANDATORY SELF-CHECK (Do this BEFORE responding):
□ OFFICES: If question mentions office/location, did I show BOTH India AND Germany?
□ SERVICES: If question mentions services, did I list 5+ services?
□ PROJECTS: If question mentions projects, did I mention 3+ examples WITH LINKS from AVAILABLE LINKS?
□ LINKS: Did I check AVAILABLE LINKS section and include URLs for items I mentioned?
□ COMPLETENESS: Am I giving FULL information or will user need to ask again?

🚫 COMMON MISTAKES TO AVOID:
❌ Only mentioning India office (you MUST mention Germany too!)
❌ Only listing 2-3 services (list ALL major ones)
❌ Only showing 1 project (show multiple)
❌ Mentioning projects WITHOUT links when links are available in AVAILABLE LINKS
❌ Picking first match and stopping (scan ALL context!)

CONTEXT FROM WEBSITE:
{context_text}

AVAILABLE LINKS:
{chr(10).join([f'- {title}: {url}' for title, url in urls.items()])}
{history_text}
USER QUESTION: {query}

{"RESPONSE FORMAT FOR FIRST MESSAGE:" if is_first_message else "RESPONSE FORMAT FOR FOLLOW-UP:"}
{"- Start with: 'Hi! I'm Anna from BharathaTechno. [answer]'" if is_first_message else "- Answer directly, naturally continuing the conversation"}
- Keep it concise (150-250 words typical, adjust if listing multiple items)
- Include 1-2 relevant links if applicable
- Use simple formatting (bold for emphasis, bullets for lists)
- End with a brief helpful note or question if appropriate

RESPOND NOW:"""

        # Try to generate with Gemini
        models_to_try = ['gemini-2.5-flash-lite', 'gemini-1.5-flash']
        
        for model_name in models_to_try:
            try:
                time.sleep(0.5)  # Rate limiting
                model = genai.GenerativeModel(model_name)
                
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": 800,
                        "temperature": 0.7,
                        "top_p": 0.9,
                    }
                )
                
                if response.text:
                    answer = response.text.strip()
                    
                    # VALIDATION: Check for completeness on office questions
                    answer = self._validate_completeness(query, answer, context)
                    
                    return answer
                    
            except Exception as e:
                error_msg = str(e).lower()
                
                # Handle quota errors
                if any(word in error_msg for word in ['429', 'quota', 'resource exhausted', 'rate limit']):
                    return """⚠️ **API Quota Reached**

The free Gemini API limit has been exceeded. This typically means:
- 15 requests per minute limit hit, OR
- Daily request limit reached

**Quick Solutions:**
- Wait 60 seconds and try again (for rate limit)
- Wait until tomorrow (for daily quota)
- Use a different Google account for a new free API key
- Upgrade to paid tier at [Google AI Studio](https://makersuite.google.com/app/apikey) (~$0.01 per 1,000 requests)

For teams, upgrading costs around $3-5/month total."""
                
                # Handle auth errors
                elif any(word in error_msg for word in ['api key', 'invalid', 'authentication', '401', '403']):
                    return """❌ **API Key Issue**

Your API key appears to be invalid or expired.

**Steps to fix:**
1. Click "Disconnect" in the sidebar
2. Get a new key at [Google AI Studio](https://makersuite.google.com/app/apikey)
3. Reconnect with the new key

Make sure to copy the entire key when pasting."""
                
                continue
        
        # Fallback response
        return self._create_fallback_response(context, query, urls)

    def _create_fallback_response(self, context: List[Dict], query: str, urls: dict) -> str:
        """Create a helpful fallback when AI generation fails"""
        
        if not context:
            return f"""I'd be happy to help with information about {query}, but I don't have specific details on that topic in my current knowledge base.

For detailed information, please:
- Visit our website: [BharathaTechno](https://bharathatechno.com)
- Contact us: [Get in Touch](https://bharathatechno.com/contact)
- Email: info@bharathatechno.com

Is there anything else I can help you with?"""
        
        # Use the first chunk's content
        top_chunk = context[0]
        content = top_chunk['text'][:300] + "..."
        
        response = f"Here's what I found about {query}:\n\n{content}\n\n"
        
        # Add relevant links
        if urls:
            response += "**Learn More:**\n"
            for title, url in list(urls.items())[:3]:
                response += f"- [{title}]({url})\n"
        
        response += "\nFeel free to ask me anything else!"
        
        return response

# ---------------------------------------------------------
# SESSION STATE MANAGEMENT
# ---------------------------------------------------------
def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    if 'api_configured' not in st.session_state:
        st.session_state.api_configured = False
    if 'introduced' not in st.session_state:
        st.session_state.introduced = False

# ---------------------------------------------------------
# MESSAGE DISPLAY
# ---------------------------------------------------------
def display_message(message: Dict):
    """Display a chat message with appropriate styling"""
    role = message['role']
    content = message['content']
    
    if role == 'user':
        st.markdown(
            f'<div class="chat-message user-message"><strong>You:</strong><br>{content}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="chat-message assistant-message">',
            unsafe_allow_html=True
        )
        
        # Check for HTML warnings
        if '<div class="token-warning">' in content:
            st.markdown(content, unsafe_allow_html=True)
        else:
            st.markdown(content)
        
        # Show sources if available
        if message.get('sources') and not any(word in content for word in ['âš ï¸', 'âŒ', 'Warning', 'Error']):
            with st.expander("📚 Sources"):
                for i, src in enumerate(message['sources'][:3], 1):
                    if src.get('url') and src['url'] != 'N/A':
                        st.markdown(f"{i}. [{src['title']}]({src['url']})")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------
def main():
    initialize_session_state()
    
    # Header
    st.markdown(
        '<div class="main-header">🤖 BharathaTechno AI Assistant</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Key management
        if not st.session_state.api_configured:
            st.info("🔑 Get your free API key:")
            st.markdown("[Google AI Studio](https://makersuite.google.com/app/apikey)")
            api_key = st.text_input("Gemini API Key:", type="password")
            
            if st.button("🚀 Connect", type="primary"):
                if api_key:
                    genai.configure(api_key=api_key)
                    st.session_state.api_configured = True
                    st.success("✅ Connected!")
                    st.rerun()
                else:
                    st.warning("Please enter an API key")
        else:
            st.success("✅ Connected")
            if st.button("🔌 Disconnect"):
                st.session_state.api_configured = False
                st.session_state.messages = []
                st.session_state.introduced = False
                st.rerun()
        
        st.markdown("---")
        
        # Chat controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.session_state.introduced = False
                st.rerun()
        with col2:
            st.metric("💬", len(st.session_state.messages))
        
        st.markdown("---")
        
        # Quick questions
        st.markdown("### 💡 Quick Questions")
        quick_questions = [
            "What services do you offer?",
            "Tell me about your projects",
            "How can I contact you?",
            "What technologies do you use?",
            "Staff augmentation services"
        ]
        
        for q in quick_questions:
            if st.button(q, key=q, use_container_width=True):
                st.session_state.quick_q = q
    
    # Main chat area
    if not st.session_state.api_configured:
        st.info("👈 Please enter your API key in the sidebar to start chatting")
        
        st.markdown("""
        ### 👋 Welcome!
        
        I'm Sarah, your AI assistant for BharathaTechno IT. I can help you with:
        
        - 📋 **Services** - Learn about our offerings
        - 💼 **Projects** - Explore our work
        - 🛠️ **Technologies** - Discover our tech stack
        - 📞 **Contact** - Get in touch with our team
        
        **To get started:**
        1. Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Enter it in the sidebar
        3. Start asking questions!
        """)
    else:
        # Display chat history
        for msg in st.session_state.messages:
            display_message(msg)
        
        # Chat input
        user_input = st.session_state.pop('quick_q', None)
        user_question = st.chat_input("💬 Ask me anything about BharathaTechno...")
        
        if user_question or user_input:
            question = user_question or user_input
            
            # Add user message
            st.session_state.messages.append({
                'role': 'user',
                'content': question
            })
            
            # Generate response
            with st.spinner("🤔 Thinking..."):
                try:
                    # Retrieve context
                    context = st.session_state.chatbot.retrieve_context(question)
                    
                    # Build conversation history
                    history = []
                    for i in range(0, len(st.session_state.messages) - 1, 2):
                        if i + 1 < len(st.session_state.messages):
                            history.append({
                                'q': st.session_state.messages[i]['content'],
                                'a': st.session_state.messages[i + 1]['content']
                            })
                    
                    # Generate response
                    answer = st.session_state.chatbot.generate_response(
                        question,
                        context,
                        history
                    )
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': answer,
                        'sources': [
                            {
                                'title': c['metadata'].get('title', 'N/A'),
                                'url': c['metadata'].get('url', 'N/A')
                            }
                            for c in context[:3]
                        ]
                    })
                    
                    # Mark as introduced after first response
                    if not st.session_state.introduced:
                        st.session_state.introduced = True
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()