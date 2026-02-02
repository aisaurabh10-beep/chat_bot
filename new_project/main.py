import os
import pickle
import logging
import warnings
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import faiss
from sentence_transformers import SentenceTransformer

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['GRPC_VERBOSITY'] = 'ERROR'
logging.getLogger('google.generativeai').setLevel(logging.ERROR)

# Configuration
DB_PATH = "vector_db_faiss"
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Set this env var!

GOOGLE_API_KEY="AIzaSyBBDrDHerfWSqTb4eIFe6B5thr08P6FYTE"

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# ---------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

class Source(BaseModel):
    title: str
    url: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]

# ---------------------------------------------------------
# CHATBOT LOGIC (Extracted from optimized_bot.py)
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
            recent_history = history[-6:] # Last few messages
            history_text = "\nRECENT CONVERSATION:\n"
            for msg in recent_history:
                role_label = "User" if msg.role == "user" else "Assistant"
                history_text += f"{role_label}: {msg.content}\n"

        is_first_message = len(history) == 0

        # Prompt construction (Identical to optimized_bot.py)
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
        models_to_try = ['gemini-2.5-flash-lite', 'gemini-1.5-flash']
        
        for model_name in models_to_try:
            try:
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

# CORS Configuration - Crucial for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Next.js URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not chatbot.model:
        raise HTTPException(status_code=503, detail="Chatbot resources not loaded")
    
    # 1. Retrieve context
    context = chatbot.retrieve_context(request.message)
    
    # 2. Generate response
    answer = chatbot.generate_response(request.message, context, request.history)
    
    # 3. Extract sources
    sources = [
        Source(title=c['metadata'].get('title', 'N/A'), url=c['metadata'].get('url', 'N/A'))
        for c in context[:3]
    ]
    
    return ChatResponse(answer=answer, sources=sources)

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": chatbot.model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)