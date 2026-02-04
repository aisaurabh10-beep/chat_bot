"""
PHASE 3: VECTOR DATABASE & EMBEDDINGS (HYBRID: WEBSITE + FILES)
====================================================================
This version uses FAISS instead of ChromaDB to avoid protobuf conflicts.
FAISS is lighter and has no protobuf dependency!

1. Loads cleaned data from 'processed_data'
2. Loads external files from 'external_data' (PDF, DOCX, TXT)
3. Splits content into optimal chunks
4. Creates embeddings using Sentence Transformers (CPU-friendly)
5. Stores in FAISS vector database
"""

import json
import os
import pickle
import glob
from datetime import datetime
from typing import List, Dict
import re
import warnings
from pypdf import PdfReader
from docx import Document

warnings.filterwarnings('ignore')

# Vector database - FAISS (no protobuf!)
import faiss
import numpy as np

# Embeddings
from sentence_transformers import SentenceTransformer

class VectorDatabaseBuilder:
    def __init__(self, cleaned_data_file: str):
        """Initialize the vector database builder"""
        self.cleaned_data_file = cleaned_data_file
        self.cleaned_data = []
        self.chunks = []
        self.index = None
        self.embedding_model = None
        
        # Configuration
        self.chunk_size = 600
        self.chunk_overlap = 100
        self.db_path = "vector_db_faiss"
        self.external_dir = "external_data"  # Folder for local files
        
        # CPU-friendly settings
        self.embedding_batch_size = 16
        
        # Statistics
        self.stats = {
            'website_pages': 0,
            'external_files': 0,
            'total_chunks': 0,
            'avg_chunk_size': 0,
            'embedding_dimension': 0
        }
    
    def load_cleaned_data(self):
        """Load cleaned data from Phase 1 (Website Data)"""
        print(f"📂 Loading cleaned data from: {self.cleaned_data_file}")
        
        try:
            with open(self.cleaned_data_file, 'r', encoding='utf-8') as f:
                self.cleaned_data = json.load(f)
            
            self.stats['website_pages'] = len(self.cleaned_data)
            print(f"✅ Loaded {self.stats['website_pages']} website pages")
            
            # Process website data immediately
            print("   Chunking website data...")
            for page in self.cleaned_data:
                metadata = {
                    'page_id': page['page_id'],
                    'title': page['title'],
                    'url': page['url'],
                    'source': 'website'
                }
                page_chunks = self.chunk_text(page['content'], metadata)
                self.chunks.extend(page_chunks)

        except Exception as e:
            print(f"⚠️ Could not load website data: {e}")

    def load_external_files(self):
        """Load local PDF, DOCX, and TXT files"""
        print(f"\n📂 Scanning '{self.external_dir}' for local files...")
        
        if not os.path.exists(self.external_dir):
            os.makedirs(self.external_dir)
            print(f"   Created empty folder '{self.external_dir}'. Put your files here!")
            return
        
        # Find all supported files
        files = []
        for ext in ['*.pdf', '*.docx', '*.txt']:
            files.extend(glob.glob(os.path.join(self.external_dir, ext)))
            
        self.stats['external_files'] = len(files)
        print(f"   Found {len(files)} documents.")

        for file_path in files:
            self.process_single_file(file_path)

    def process_single_file(self, file_path: str):
        """Extract text from a single file and chunk it"""
        filename = os.path.basename(file_path)
        text = ""
        
        try:
            # Extract Text based on extension
            if file_path.endswith('.pdf'):
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text += (page.extract_text() or "") + "\n"
            
            elif file_path.endswith('.docx'):
                doc = Document(file_path)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            
            elif file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

            # Clean generic whitespace for better chunking
            text = re.sub(r'\s+', ' ', text).strip()
            
            if not text:
                print(f"   ⚠️ Skipping empty file: {filename}")
                return

            # Create Metadata
            metadata = {
                'page_id': f"doc_{filename}",
                'title': filename,
                'url': 'External Document', 
                'source': 'local_file'
            }
            
            # Chunk
            file_chunks = self.chunk_text(text, metadata)
            self.chunks.extend(file_chunks)
            print(f"   + Added {filename} ({len(file_chunks)} chunks)")
            
        except Exception as e:
            print(f"   ❌ Error reading {filename}: {e}")

    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """Split text into overlapping chunks"""
        chunks = []
        
        # If text looks like a real document (no clear double newlines), 
        # we might need to be careful. But for now, we use your original logic 
        # which splits by paragraphs and then sentences.
        paragraphs = text.split('\n\n')
        
        # Fallback for PDFs that might come out as one block
        if len(paragraphs) < 2 and len(text) > self.chunk_size:
             # Treat the whole text as one 'paragraph' to enter the splitting logic
             paragraphs = [text]

        current_chunk = ""
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = len(para)
            
            if para_size > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': metadata.copy()
                    })
                    current_chunk = ""
                    current_size = 0
                
                # Split huge paragraphs by sentence
                sentences = re.split(r'[.!?]+', para)
                temp_chunk = ""
                temp_size = 0
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    sentence_size = len(sentence) + 1
                    
                    if temp_size + sentence_size > self.chunk_size:
                        if temp_chunk:
                            chunks.append({
                                'text': temp_chunk.strip(),
                                'metadata': metadata.copy()
                            })
                        temp_chunk = sentence + ". "
                        temp_size = sentence_size
                    else:
                        temp_chunk += sentence + ". "
                        temp_size += sentence_size
                
                if temp_chunk:
                    chunks.append({
                        'text': temp_chunk.strip(),
                        'metadata': metadata.copy()
                    })
            
            elif current_size + para_size > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': metadata.copy()
                    })
                
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self.chunk_overlap:].strip()
                    current_chunk = overlap_text + "\n\n" + para
                    current_size = len(current_chunk)
                else:
                    current_chunk = para
                    current_size = para_size
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_size += para_size + 2
        
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': metadata.copy()
            })
        
        return chunks
    
    def initialize_embedding_model(self):
        """Initialize the sentence transformer model (CPU-optimized)"""
        print("\n🤖 Initializing embedding model (CPU mode)...")
        print("   Model: all-MiniLM-L6-v2")
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.embedding_model.eval()
        
        self.stats['embedding_dimension'] = self.embedding_model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded! Dimension: {self.stats['embedding_dimension']}")

    def create_vector_database(self):
        """Create FAISS vector database"""
        print("\n🗄️  Creating FAISS vector database...")
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
        self.index = faiss.IndexFlatIP(self.stats['embedding_dimension'])

    def add_chunks_to_database(self):
        """Generate embeddings and add chunks to database"""
        if not self.chunks:
            print("❌ No data to index!")
            return

        print(f"\n📊 Generating embeddings for {len(self.chunks)} chunks...")
        
        # Extract all texts
        texts = [chunk['text'] for chunk in self.chunks]
        
        # Generate all embeddings
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.embedding_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Save index and metadata
        print(f"\n💾 Saving database to {self.db_path}...")
        faiss.write_index(self.index, os.path.join(self.db_path, 'index.faiss'))
        
        # Add index numbers to metadata before saving
        for i, chunk in enumerate(self.chunks):
            chunk['metadata']['chunk_index'] = i
            chunk['metadata']['total_chunks'] = len(self.chunks)

        with open(os.path.join(self.db_path, 'chunks.pkl'), 'wb') as f:
            pickle.dump(self.chunks, f)
        
        self.stats['total_chunks'] = len(self.chunks)
        print(f"✅ Saved {self.stats['total_chunks']} chunks.")

    def test_retrieval(self, test_queries: List[str], n_results: int = 3):
        """Test the retrieval system"""
        print("\n🔍 Testing retrieval system...")
        print("=" * 60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Test Query {i}: '{query}'")
            
            query_embedding = self.embedding_model.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True
            )
            
            distances, indices = self.index.search(query_embedding, n_results)
            
            for j, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    meta = chunk['metadata']
                    print(f"   {j}. [{meta.get('source','?')}] {meta.get('title', 'N/A')[:40]} (Score: {dist:.3f})")
                    print(f"      Preview: {chunk['text'][:100]}...")

    def generate_report(self):
        """Generate report"""
        report_file = f'phase2_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
        print(f"\n📄 Report saved to: {report_file}")


def main():
    """Main execution"""
    print("\n" + "=" * 60)
    print("🚀 PHASE 2: VECTOR DATABASE (HYBRID: WEB + FILES)")
    print("=" * 60 + "\n")
    
    # 1. SETUP PATHS
    cleaned_dir = 'processed_data'  # YOUR CUSTOM FOLDER
    
    if not os.path.exists(cleaned_dir):
        print(f"❌ Error: '{cleaned_dir}' directory not found!")
        return
    
    json_files = [f for f in os.listdir(cleaned_dir) 
                  if f.startswith('clean_data') and f.endswith('.json')]
    
    if not json_files:
        print(f"❌ Error: No cleaned data files found in {cleaned_dir}!")
        return
    
    json_files.sort(reverse=True)
    cleaned_data_file = os.path.join(cleaned_dir, json_files[0])
    
    # 2. INITIALIZE
    builder = VectorDatabaseBuilder(cleaned_data_file)
    
    # 3. LOAD DATA (WEB + EXTERNAL)
    builder.load_cleaned_data()    # Load website data from JSON
    builder.load_external_files()  # Load PDF/Docx from external_data/
    
    # 4. BUILD DB
    builder.initialize_embedding_model()
    # Note: create_chunks is now integrated into the loading functions
    builder.create_vector_database()
    builder.add_chunks_to_database()
    
    # 5. TEST
    print("\n" + "=" * 60)
    test_queries = [
        "What services does the company offer?",
        "Company information and background"
    ]
    builder.test_retrieval(test_queries, n_results=3)
    builder.generate_report()
    
    print("=" * 60)
    print("✅ PROCESS COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()