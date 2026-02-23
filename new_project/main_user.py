import os
import json
import pickle
import logging
import warnings
import re
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Dict

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai
import faiss
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')
os.environ['GRPC_VERBOSITY'] = 'ERROR'
logging.getLogger('google.generativeai').setLevel(logging.ERROR)

DB_PATH = 'vector_db_faiss'
LOGS_DIR = 'user_chat_logs'
MIN_RETRIEVAL_SCORE = 0.20
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    user_id: str = Field(..., description='Unique identifier for the user (e.g., email or UUID)')
    message: str
    history: List[ChatMessage] = []


class Source(BaseModel):
    title: str
    url: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str


class ChatLogger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def _sanitize_filename(self, user_id: str) -> str:
        safe_id = re.sub(r'[^a-zA-Z0-9_\-@.]', '_', user_id)
        return f'{safe_id}.json'

    def log_interaction(self, user_id: str, question: str, answer: str, sources: List[Source]):
        filename = self._sanitize_filename(user_id)
        filepath = os.path.join(self.log_dir, filename)

        entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'question': question,
            'answer': answer,
            'sources': [s.dict() for s in sources],
        }

        data = []
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = []

        data.append(entry)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


chat_logger = ChatLogger(LOGS_DIR)


class RAGChatbot:
    def __init__(self):
        self.index = None
        self.chunks = None
        self.model = None

    def load_resources(self):
        self.index = faiss.read_index(os.path.join(DB_PATH, 'index.faiss'))
        with open(os.path.join(DB_PATH, 'chunks.pkl'), 'rb') as f:
            self.chunks = pickle.load(f)
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.model.eval()

    def _query_variants(self, query: str) -> List[str]:
        compact = re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9\s]', ' ', query)).strip()
        keywords = " ".join([w for w in compact.split() if len(w) > 3][:8])
        variants = [query]
        if compact and compact.lower() != query.lower():
            variants.append(compact)
        if keywords and keywords.lower() not in [v.lower() for v in variants]:
            variants.append(keywords)
        return variants[:3]

    def _source_label(self, metadata: Dict) -> str:
        title = metadata.get('title', 'Unknown Source')
        source = metadata.get('source', 'website')
        doc_type = metadata.get('doc_type', '')
        page_number = metadata.get('page_number')
        if source == 'local_file' and doc_type == 'pdf' and page_number:
            return f"{title} (PDF page {page_number})"
        return title

    def retrieve_context(self, query: str, top_k: int = 18) -> List[Dict]:
        if not self.model:
            return []
        collected = []
        seen = set()
        for variant in self._query_variants(query):
            vec = self.model.encode([variant], normalize_embeddings=True)
            scores, idxs = self.index.search(vec, top_k)
            for score, idx in zip(scores[0], idxs[0]):
                if float(score) < MIN_RETRIEVAL_SCORE:
                    continue
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    sig = chunk['text'][:200]
                    if sig in seen:
                        continue
                    item = {
                        'text': chunk['text'],
                        'metadata': chunk['metadata'],
                        'score': float(score)
                    }
                    collected.append(item)
                    seen.add(sig)

        collected.sort(key=lambda x: x['score'], reverse=True)
        return collected[:top_k]

    def generate_response(self, query: str, context: List[Dict], history: List[ChatMessage]) -> str:
        context_text = '\n\n'.join([
            f"REF: [{i+1}] {self._source_label(c['metadata'])}\nURL: {c['metadata'].get('url', 'N/A')}\nCONTENT: {c['text']}"
            for i, c in enumerate(context)
        ]) if context else 'No relevant information found.'

        urls = {}
        for c in context:
            url = c['metadata'].get('url', '')
            title = c['metadata'].get('title', '')
            if url and url != 'N/A' and title:
                urls[title] = url

        history_text = ''
        if history:
            recent_history = history[-10:]
            history_text = '\nRECENT CONVERSATION:\n'
            for msg in recent_history:
                role = 'User' if msg.role == 'user' else 'Assistant'
                history_text += f'{role}: {msg.content}\n'

        prompt = f"""You are a knowledge-grounded assistant for Sant Rampal Ji Maharaj website content.

CRITICAL RULES:
1. Answer ONLY from the provided CONTEXT.
2. Do NOT use outside/world knowledge, assumptions, or generic explanations.
3. If the answer is missing in CONTEXT, reply exactly: "I don't have this information in the provided data."
4. Response style: slightly descriptive but precise (short explanation with key points).
5. Synthesize from multiple relevant context entries, not just one line.
6. Every factual statement should be supported by context.
7. End with a "References" section and cite source refs like [1], [2]. For PDF content include page number if present.
8. If relevant links are present, include them in markdown format.
9. For follow-up questions, continue naturally without repeating introductions.
9. Never present uncertain content as fact.

CONTEXT FROM WEBSITE:
{context_text}

AVAILABLE LINKS:
{chr(10).join([f'- {title}: {url}' for title, url in urls.items()])}
{history_text}
USER QUESTION: {query}

RESPOND NOW:"""

        for model_name in ['gemini-2.5-flash-lite', 'gemini-1.5-flash']:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config={
                        'max_output_tokens': 700,
                        'temperature': 0.3,
                        'top_p': 0.8,
                    }
                )
                if response.text:
                    return response.text.strip()
            except Exception:
                continue

        if not context:
            return "I don't have this information in the provided data."

        top_chunk = context[0]
        content = top_chunk['text'][:300] + '...'
        ref = self._source_label(top_chunk['metadata'])
        result = f"Here is what I found in the provided data:\n\n{content}\n\n"
        result += f"References:\n- {ref}\n"
        if urls:
            result += '**Relevant Links:**\n'
            for title, url in list(urls.items())[:3]:
                result += f'- [{title}]({url})\n'
        return result


chatbot = RAGChatbot()


@asynccontextmanager
async def lifespan(app: FastAPI):
    chatbot.load_resources()
    yield


app = FastAPI(title='Sant Rampal Ji Maharaj Knowledge API', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.post('/chat', response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    if not chatbot.model:
        raise HTTPException(status_code=503, detail='Chatbot resources not loaded')

    context = chatbot.retrieve_context(request.message)
    answer = chatbot.generate_response(request.message, context, request.history)

    sources = [
        Source(
            title=chatbot._source_label(c['metadata']),
            url=c['metadata'].get('url', 'N/A')
        )
        for c in context[:5]
    ]

    background_tasks.add_task(
        chat_logger.log_interaction,
        request.user_id,
        request.message,
        answer,
        sources,
    )

    return ChatResponse(answer=answer, session_id=request.user_id)


@app.get('/health')
async def health_check():
    return {'status': 'ok', 'model_loaded': chatbot.model is not None}


if __name__ == '__main__':
    import uvicorn
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    uvicorn.run(app, host='0.0.0.0', port=8010)
