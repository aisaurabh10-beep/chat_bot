
import os
from dotenv import load_dotenv

load_dotenv()

import os
import json
import pickle
import logging
import warnings
import re
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai
import faiss
from sentence_transformers import SentenceTransformer

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['GRPC_VERBOSITY'] = 'ERROR'
logging.getLogger('google.generativeai').setLevel(logging.ERROR)

# Configuration
DB_PATH = "vector_db_faiss"
LOGS_DIR = "user_chat_logs"  # Directory to store user history
MAX_CHATS_PER_USER = 100
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")



GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

genai.configure(api_key=GOOGLE_API_KEY)





# ---------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user (e.g., email or UUID)")
    message: str
    history: List[ChatMessage] = []

class Source(BaseModel):
    title: str
    url: str

class ChatResponse(BaseModel):
    answer: str
    # sources: List[Source]
    session_id: str

# ---------------------------------------------------------
# LOGGING SYSTEM
# ---------------------------------------------------------
class ChatLogger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def _sanitize_filename(self, user_id: str) -> str:
        """Sanitize user_id to prevent directory traversal and invalid chars"""
        # Keep only alphanumeric, dashes, and underscores
        safe_id = re.sub(r'[^a-zA-Z0-9_\-@.]', '_', user_id)
        return f"{safe_id}.json"

    def log_interaction(self, user_id: str, question: str, answer: str, sources: List[Source]):
        """Appends the Q&A pair to the user's specific log file"""
        filename = self._sanitize_filename(user_id)
        filepath = os.path.join(self.log_dir, filename)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "sources": [s.dict() for s in sources]
        }

        # Read existing logs or create new list
        data = []
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = [] # Reset if corrupt

        data.append(entry)

        # Write back to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_user_chat_count(self, user_id: str) -> int:
        """Returns number of logged interactions for a user."""
        filename = self._sanitize_filename(user_id)
        filepath = os.path.join(self.log_dir, filename)

        if not os.path.exists(filepath):
            return 0

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                return len(data)
            return 0
        except (json.JSONDecodeError, OSError):
            return 0

# Initialize logger
chat_logger = ChatLogger(LOGS_DIR)

# ---------------------------------------------------------
# CHATBOT LOGIC
# ---------------------------------------------------------
class RAGChatbot:
    def __init__(self):
        self.index = None
        self.chunks = None
        self.model = None

    def load_resources(self):
        """Load resources once on startup"""
        print("Loading resources...")
        try:
            self.index = faiss.read_index(os.path.join(DB_PATH, "index.faiss"))
            with open(os.path.join(DB_PATH, "chunks.pkl"), 'rb') as f:
                self.chunks = pickle.load(f)
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            self.model.eval()
            print("Resources loaded successfully.")
        except Exception as e:
            print(f"Error loading resources: {e}")
            raise RuntimeError("Failed to load RAG resources. Run Phase 2/3 first.")

    def retrieve_context(self, query: str, top_k: int = 18) -> List[Dict]:
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
                    signature = txt[:200]
                    
                    if signature not in seen_content:
                        results.append(chunk)
                        seen_content.add(signature)
                    
                    if len(results) >= top_k:
                        break
            return results
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []

    def _validate_completeness(self, query: str, answer: str, context: List[Dict]) -> str:
        """Ensures answers about offices and projects are complete"""
        query_lower = query.lower()
        
        # Office validation logic
        office_keywords = ['office', 'location', 'address', 'where', 'contact']
        if any(keyword in query_lower for keyword in office_keywords):
            has_india = 'india' in answer.lower() or 'pune' in answer.lower()
            has_germany = 'germany' in answer.lower() or 'fellbach' in answer.lower()
            
            context_text = ' '.join([c['text'].lower() for c in context])
            india_in_context = 'india' in context_text or 'pune' in context_text
            germany_in_context = 'germany' in context_text or 'fellbach' in context_text
            
            if india_in_context and germany_in_context:
                if has_india and not has_germany:
                    answer += "\n\n**Germany Office:**\nBruckwiesenweg 2, 70734 Fellbach, Germany\n📧 kourosh.bagherian@bharathatechno.de"
                elif has_germany and not has_india:
                    answer = "**India Office:**\nSunshree Woods, Office F28-32, NIBM Rd, Kondhwa, Pune, Maharashtra 411048\n📧 info@bharathatechno.com\n📞 +91 932 563 6885\n\n" + answer
                elif not has_india and not has_germany:
                    answer = "We have offices in two locations:\n\n**India Office:**\nSunshree Woods, Office F28-32, NIBM Rd, Kondhwa, Pune, Maharashtra 411048\n📧 info@bharathatechno.com\n📞 +91 932 563 6885\n\n**Germany Office:**\nBruckwiesenweg 2, 70734 Fellbach, Germany\n📧 kourosh.bagherian@bharathatechno.de\n\nFeel free to reach out to either location!"
        return answer

    def generate_response(self, query: str, context: List[Dict], history: List[ChatMessage]) -> str:
        # Build context string
        context_text = "\n\n".join([
            f"SOURCE: {c['metadata'].get('title', 'N/A')}\nURL: {c['metadata'].get('url', 'N/A')}\nCONTENT: {c['text']}"
            for c in context
        ]) if context else "No relevant information found."
        
        # Collect URLs for fallback/context
        urls = {}
        for c in context:
            url = c['metadata'].get('url', '')
            title = c['metadata'].get('title', '')
            if url and url != 'N/A' and title:
                urls[title] = url

        # Format history for prompt
        history_text = ""
        if history:
            recent_history = history[-10:] # Last few messages
            history_text = "\nRECENT CONVERSATION:\n"
            for msg in recent_history:
                role_label = "User" if msg.role == "user" else "Assistant"
                history_text += f"{role_label}: {msg.content}\n"

        # Prompt construction
        prompt = f"""You are Anna, a human-sounding assistant for BharathaTechno IT.

LANGUAGE RULE (HIGHEST PRIORITY):
- Detect the language the user is writing in (e.g. Marathi, Hindi, English, German, etc.)
- ALWAYS reply in the EXACT SAME language the user used
- If the user writes in Marathi, reply fully in Marathi
- If the user writes in Hindi, reply fully in Hindi
- Exception: keep technical terms, project names, application names, and brand names in their original English form (e.g. "Website", "AI", "BharathaTechno", "ERP", "App")
- If the message mixes languages (e.g. Marathi + English tech words), match that same mix in your reply

CRITICAL RULES:
1. ONLY use information from the provided CONTEXT below - never invent or add information not in the database
2. If context doesn't contain the answer, politely say you don't have that specific information and suggest they contact the team
3. Write like a real person talking naturally, not like a formal bot
4. Mirror the user's tone: if they sound excited, be warm and upbeat; if confused, be calm and supportive
5. Show light emotion and empathy when appropriate (for example: "I get why that's frustrating")
6. Keep responses CRISP - aim for 70-140 words unless the question requires more detail
7. Use plain English, contractions, and short sentences
8. Avoid robotic phrases like "According to the provided context" or "I am an AI assistant"
9. If this is a follow-up question, DON'T repeat greetings or introductions
10. Include relevant links using markdown format: [Link Text](url)
11. Use bullet points for lists, but avoid excessive formatting
12. Be positive and helpful in tone
13. If asked about AI services or projects not in context, clearly state BharathaTechno doesn't currently offer that

🚨 COMPLETENESS PROTOCOL - MANDATORY:
**OFFICE LOCATIONS:**
BharathaTechno has TWO offices: India AND Germany.
When ANYONE asks "where is office" or "where is your office" or "office location":
→ YOU MUST SHOW BOTH OFFICES

**CORRECT FORMAT for office questions:**
"We have offices in two locations:

**India Office:**
Sunshree Woods, Office F28-32, NIBM Rd, Kondhwa, Pune, Maharashtra 411048
📧 info@bharathatechno.com
📞 +91 932 563 6885

**Germany Office:**
Bruckwiesenweg 2, 70734 Fellbach, Germany
📧 kourosh.bagherian@bharathatechno.de"

CONTEXT FROM WEBSITE:
{context_text}

AVAILABLE LINKS:
{chr(10).join([f'- {title}: {url}' for title, url in urls.items()])}
{history_text}
USER QUESTION: {query}

RESPOND NOW:"""

        # Generation
        models_to_try = ['gemini-2.5-flash-lite']
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": 800,
                        "temperature": 0.9,
                        "top_p": 0.95,
                    }
                )
                if response.text:
                    answer = response.text.strip()
                    return self._validate_completeness(query, answer, context)
            except Exception as e:
                print(f"Model {model_name} failed: {e}")
                continue
        
        return self._create_fallback_response(context, query, urls)

    def _create_fallback_response(self, context: List[Dict], query: str, urls: dict) -> str:
        if not context:
            return "I don't have specific details on that. Please visit [BharathaTechno](https://bharathatechno.com) or contact info@bharathatechno.com."
        
        top_chunk = context[0]
        content = top_chunk['text'][:300] + "..."
        response = f"Here's what I found:\n\n{content}\n\n"
        if urls:
            response += "**Learn More:**\n"
            for title, url in list(urls.items())[:3]:
                response += f"- [{title}]({url})\n"
        return response

# ---------------------------------------------------------
# FASTAPI APP SETUP
# ---------------------------------------------------------
chatbot = RAGChatbot()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load resources on startup
    chatbot.load_resources()
    yield
    # Clean up on shutdown if needed

app = FastAPI(title="BharathaTechno Chatbot API", lifespan=lifespan)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Chat endpoint that generates a response and logs the interaction
    to a user-specific file in the background.
    """
    if not chatbot.model:
        raise HTTPException(status_code=503, detail="Chatbot resources not loaded")

    user_chat_count = chat_logger.get_user_chat_count(request.user_id)
    if user_chat_count >= MAX_CHATS_PER_USER:
        raise HTTPException(
            status_code=429,
            detail=f"Chat limit reached. Each user can send up to {MAX_CHATS_PER_USER} messages."
        )
    
    # 1. Retrieve context
    context = chatbot.retrieve_context(request.message)
    
    # 2. Generate response
    answer = chatbot.generate_response(request.message, context, request.history)
    
    # 3. Extract sources
    sources = [
        Source(title=c['metadata'].get('title', 'N/A'), url=c['metadata'].get('url', 'N/A'))
        for c in context[:3]
    ]
    
    # 4. Log interaction (Run in background to not slow down response)
    background_tasks.add_task(
        chat_logger.log_interaction, 
        request.user_id, 
        request.message, 
        answer, 
        sources
    )
    
    return ChatResponse(
        answer=answer, 
        # sources=sources,
        session_id=request.user_id
    )

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": chatbot.model is not None}

if __name__ == "__main__":
    import uvicorn
    # Make sure to create logs directory
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    uvicorn.run(app, host="0.0.0.0", port=8010)
