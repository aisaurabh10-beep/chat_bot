import os
import pickle
import warnings
import time
import logging
from typing import List, Dict
import re

import streamlit as st
import google.generativeai as genai
import faiss
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')
os.environ['GRPC_VERBOSITY'] = 'ERROR'
logging.getLogger('google.generativeai').setLevel(logging.ERROR)
MIN_RETRIEVAL_SCORE = 0.20

st.set_page_config(
    page_title='Sant Rampal Ji Maharaj Assistant',
    page_icon='🕉️',
    layout='wide'
)

st.markdown('''
<style>
    .main-header {
        font-size: 2.0rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .user-message {
        background: #eef5ff;
        border-left: 4px solid #2b6cb0;
    }
    .assistant-message {
        background: #f7fafc;
        border-left: 4px solid #2f855a;
    }
</style>
''', unsafe_allow_html=True)


@st.cache_resource
def load_resources():
    try:
        db_path = 'vector_db_faiss'
        index = faiss.read_index(os.path.join(db_path, 'index.faiss'))
        with open(os.path.join(db_path, 'chunks.pkl'), 'rb') as f:
            chunks = pickle.load(f)
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        model.eval()
        return index, chunks, model
    except Exception as e:
        st.error(f'Error loading resources: {e}')
        return None, None, None


class RAGChatbot:
    def __init__(self):
        self.index, self.chunks, self.model = load_resources()
        self.initialized = self.index is not None

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
                    collected.append({
                        'text': chunk['text'],
                        'metadata': chunk['metadata'],
                        'score': float(score)
                    })
                    seen.add(sig)
        collected.sort(key=lambda x: x['score'], reverse=True)
        return collected[:top_k]

    def generate_response(self, query: str, context: List[Dict], history: List[Dict]) -> str:
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
            recent_history = history[-4:]
            history_text = '\nRECENT CONVERSATION:\n'
            for h in recent_history:
                history_text += f"User: {h['q']}\nAssistant: {h['a'][:120]}...\n"

        is_first_message = len(history) == 0

        prompt = f"""You are a knowledge-grounded assistant for Sant Rampal Ji Maharaj website content.

CRITICAL RULES:
1. Answer ONLY from the provided CONTEXT.
2. Do NOT use outside/world knowledge, assumptions, or generic explanations.
3. If the answer is missing in CONTEXT, reply exactly: "I don't have this information in the provided data."
4. Response style: slightly descriptive but precise (short explanation with key points).
5. Synthesize from multiple relevant context entries, not just one line.
6. Every factual statement should be supported by context.
7. End with a "References" section and cite source refs like [1], [2]. For PDF content include page number if present.
8. If relevant links are present in AVAILABLE LINKS, include them.
9. For follow-up questions, continue naturally without repeating introductions.
10. Never present uncertain content as fact.

CONTEXT FROM WEBSITE:
{context_text}

AVAILABLE LINKS:
{chr(10).join([f'- {title}: {url}' for title, url in urls.items()])}
{history_text}
USER QUESTION: {query}

{"RESPONSE FORMAT FOR FIRST MESSAGE:" if is_first_message else "RESPONSE FORMAT FOR FOLLOW-UP:"}
{"- Start with: 'Hi, I can help with Sant Rampal Ji Maharaj website content.'" if is_first_message else "- Answer directly and clearly"}
- Keep it concise but informative
- Use bullets when useful
- End with a References section using [1], [2]
- Do not add outside information

RESPOND NOW:"""

        for model_name in ['gemini-2.5-flash-lite', 'gemini-1.5-flash']:
            try:
                time.sleep(0.3)
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
        result = f"Here is what I found in the provided data:\n\n{content}\n\n"
        result += f"References:\n- {self._source_label(top_chunk['metadata'])}\n"
        if urls:
            result += '**Relevant Links:**\n'
            for title, url in list(urls.items())[:3]:
                result += f'- [{title}]({url})\n'
        result += '\nAsk another question from the provided data if needed.'
        return result


def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    if 'api_configured' not in st.session_state:
        st.session_state.api_configured = False


def display_message(message: Dict):
    role = message['role']
    content = message['content']

    if role == 'user':
        st.markdown(f'<div class="chat-message user-message"><strong>You:</strong><br>{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-message assistant-message">', unsafe_allow_html=True)
        st.markdown(content)
        if message.get('sources'):
            with st.expander('Sources'):
                for i, src in enumerate(message['sources'][:5], 1):
                    if src.get('url') and src['url'] not in ['N/A', 'External Document']:
                        st.markdown(f'{i}. [{src["title"]}]({src["url"]})')
                    else:
                        st.markdown(f'{i}. {src["title"]}')
        st.markdown('</div>', unsafe_allow_html=True)


def main():
    initialize_session_state()

    st.markdown('<div class="main-header">Sant Rampal Ji Maharaj Assistant</div>', unsafe_allow_html=True)
    st.markdown('---')

    with st.sidebar:
        st.header('Settings')

        if not st.session_state.api_configured:
            st.info('Get your API key:')
            st.markdown('[Google AI Studio](https://makersuite.google.com/app/apikey)')
            api_key = st.text_input('Gemini API Key:', type='password')
            if st.button('Connect', type='primary'):
                if api_key:
                    genai.configure(api_key=api_key)
                    st.session_state.api_configured = True
                    st.rerun()
                else:
                    st.warning('Please enter an API key')
        else:
            st.success('Connected')
            if st.button('Disconnect'):
                st.session_state.api_configured = False
                st.session_state.messages = []
                st.rerun()

        st.markdown('---')
        if st.button('Clear Chat'):
            st.session_state.messages = []
            st.rerun()

        st.markdown('---')
        st.markdown('### Quick Questions')
        quick_questions = [
            'Who is Sant Rampal Ji Maharaj?',
            'What is true worship according to your data?',
            'Share teachings from Gita references in the data',
            'What is naam diksha according to the content?',
            'Show key points from Satsang content'
        ]
        for q in quick_questions:
            if st.button(q, key=q, use_container_width=True):
                st.session_state.quick_q = q

    if not st.session_state.api_configured:
        st.info('Please enter your API key in the sidebar to start chatting')
        st.markdown('''
### Welcome!

This assistant answers only from your indexed Sant Rampal Ji Maharaj website data.

- It does not use outside or worldly knowledge.
- If something is not in the data, it will say so.
- Source website: https://www.jagatgururampalji.org/

**To get started:**
1. Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Enter it in the sidebar
3. Start asking questions from the indexed content
''')
    else:
        for msg in st.session_state.messages:
            display_message(msg)

        user_input = st.session_state.pop('quick_q', None)
        user_question = st.chat_input('Ask from Sant Rampal Ji Maharaj data...')

        if user_question or user_input:
            question = user_question or user_input
            st.session_state.messages.append({'role': 'user', 'content': question})

            with st.spinner('Thinking...'):
                try:
                    context = st.session_state.chatbot.retrieve_context(question)
                    history = []
                    for i in range(0, len(st.session_state.messages) - 1, 2):
                        if i + 1 < len(st.session_state.messages):
                            history.append({
                                'q': st.session_state.messages[i]['content'],
                                'a': st.session_state.messages[i + 1]['content']
                            })

                    answer = st.session_state.chatbot.generate_response(question, context, history)

                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': answer,
                        'sources': [
                            {
                                'title': st.session_state.chatbot._source_label(c['metadata']),
                                'url': c['metadata'].get('url', 'N/A')
                            }
                            for c in context[:5]
                        ]
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f'An error occurred: {str(e)}')


if __name__ == '__main__':
    main()

