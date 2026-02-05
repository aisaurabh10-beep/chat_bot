import os
import json
import pickle
import logging
import warnings
import re
import time
import random
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

# ---------------------------------------------------------
# CONFIGURATION & KEY MANAGEMENT
# ---------------------------------------------------------
DB_PATH = "vector_db_faiss"
LOGS_DIR = "user_chat_logs"

# List of your Gemini API keys for rotation
LIST_OF_KEYS = [
    "AIzaSyAsqJyOfukbt3N53XLaAvE_HVNSSaS_kic",
    "AIzaSyBBDrDHerfWSqTb4eIFe6B5thr08P6FYTE",
    "AIzaSyC-8u9GvCbATHZZrHcB0l-QR1VDQ26C-1M"
]

class GeminiKeyManager:
    def __init__(self, api_keys: List[str]):
        # Initialize keys with health tracking
        self.keys = [{"key": k, "active": True, "last_used": 0} for k in api_keys]
        self.current_index = 0

    def get_available_key(self) -> str:
        """Returns an active key using round-robin logic"""
        start_index = self.current_index
        while True:
            candidate = self.keys[self.current_index]
            
            # Reactivate key if it has been cooling down for more than 2 minutes
            if not candidate["active"] and (time.time() - candidate["last_used"] > 120):
                candidate["active"] = True

            if candidate["active"]:
                key = candidate["key"]
                # Move index for next call
                self.current_index = (self.current_index + 1) % len(self.keys)
                return key

            self.current_index = (self.current_index + 1) % len(self.keys)
            
            # If we've looped through all keys and none are active
            if self.current_index == start_index:
                raise RuntimeError("All Gemini API keys are currently exhausted.")

    def mark_key_failed(self, key: str):
        """Marks a key as inactive when it hits a rate limit (HTTP 429)"""
        for k_obj in self.keys:
            if k_obj["key"] == key:
                k_obj["active"] = False
                k_obj["last_used"] = time.time()
                print(f"⚠️ Key {key[:8]}... marked inactive due to rate limiting.")
                break

# Initialize the manager
key_manager = GeminiKeyManager(LIST_OF_KEYS)

# ---------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    message: str
    history: List[ChatMessage] = []

class Source(BaseModel):
    title: str
    url: str

class ChatResponse(BaseModel):
    answer: str
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
        safe_id = re.sub(r'[^a-zA-Z0-9_\-@.]', '_', user_id)
        return f"{safe_id}.json"

    def log_interaction(self, user_id: str, question: str, answer: str, sources: List[Source]):
        filename = self._sanitize_filename(user_id)
        filepath = os.path.join(self.log_dir, filename)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "sources": [s.dict() for s in sources]
        }

        data = []
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except:
                data = []

        data.append(entry)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

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
        try:
            self.index = faiss.read_index(os.path.join(DB_PATH, "index.faiss"))
            with open(os.path.join(DB_PATH, "chunks.pkl"), 'rb') as f:
                self.chunks = pickle.load(f)
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load RAG resources: {e}")

    def retrieve_context(self, query: str, top_k: int = 18) -> List[Dict]:
        if not self.model: return []
        try:
            vec = self.model.encode([query], normalize_embeddings=True)
            dists, idxs = self.index.search(vec, top_k * 2)
            results = []
            seen_content = set()
            for idx in idxs[0]:
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    signature = chunk['text'][:200]
                    if signature not in seen_content:
                        results.append(chunk)
                        seen_content.add(signature)
                    if len(results) >= top_k: break
            return results
        except:
            return []

    def _validate_completeness(self, query: str, answer: str, context: List[Dict]) -> str:
        query_lower = query.lower()
        office_keywords = ['office', 'location', 'address', 'where', 'contact']
        if any(keyword in query_lower for keyword in office_keywords):
            has_india = any(x in answer.lower() for x in ['india', 'pune'])
            has_germany = any(x in answer.lower() for x in ['germany', 'fellbach'])
            
            if not (has_india and has_germany):
                return "We have offices in two locations:\n\n**India Office:**\nSunshree Woods, Office F28-32, NIBM Rd, Kondhwa, Pune, Maharashtra 411048\n📧 info@bharathatechno.com\n📞 +91 932 563 6885\n\n**Germany Office:**\nBruckwiesenweg 2, 70734 Fellbach, Germany\n📧 kourosh.bagherian@bharathatechno.de"
        return answer

    def generate_response(self, query: str, context: List[Dict], history: List[ChatMessage]) -> str:
        # Prompt Construction
        context_text = "\n\n".join([f"SOURCE: {c['metadata'].get('title')}\nCONTENT: {c['text']}" for c in context])
        history_text = "\n".join([f"{'User' if m.role=='user' else 'Assistant'}: {m.content}" for m in history[-5:]])
        
        prompt = f"""You are Anna, an AI assistant for BharathaTechno IT. 
        Only use this context: {context_text}
        History: {history_text}
        User: {query}"""

        # API KEY ROTATION LOGIC
        max_retries = len(LIST_OF_KEYS)
        for attempt in range(max_retries):
            current_key = key_manager.get_available_key()
            try:
                genai.configure(api_key=current_key)
                # Note: Using 1.5-flash as it is more stable for rotation than 2.5-lite-preview
                model = genai.GenerativeModel('gemini-2.5-flash-lite')
                
                response = model.generate_content(
                    prompt,
                    generation_config={"max_output_tokens": 800, "temperature": 0.7}
                )
                
                if response.text:
                    return self._validate_completeness(query, response.text.strip(), context)
            
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "quota" in error_str:
                    key_manager.mark_key_failed(current_key)
                    continue # Try next key
                else:
                    print(f"Generation error: {e}")
                    break

        return "I'm experiencing high traffic right now. Please try again in a moment."

# ---------------------------------------------------------
# FASTAPI APP SETUP
# ---------------------------------------------------------
chatbot = RAGChatbot()

@asynccontextmanager
async def lifespan(app: FastAPI):
    chatbot.load_resources()
    yield

app = FastAPI(title="BharathaTechno Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    if not chatbot.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    context = chatbot.retrieve_context(request.message)
    answer = chatbot.generate_response(request.message, context, request.history)
    
    sources = [
        Source(title=c['metadata'].get('title', 'N/A'), url=c['metadata'].get('url', 'N/A'))
        for c in context[:3]
    ]
    
    background_tasks.add_task(
        chat_logger.log_interaction, 
        request.user_id, request.message, answer, sources
    )
    
    return ChatResponse(answer=answer, session_id=request.user_id)

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": chatbot.model is not None}

if __name__ == "__main__":
    import uvicorn
    if not os.path.exists(LOGS_DIR): os.makedirs(LOGS_DIR)
    uvicorn.run(app, host="0.0.0.0", port=8010)