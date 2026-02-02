"""
PHASE 2: VECTOR DATABASE & EMBEDDINGS (FAISS VERSION - NO PROTOBUF)
====================================================================
This version uses FAISS instead of ChromaDB to avoid protobuf conflicts.
FAISS is lighter and has no protobuf dependency!

1. Loads cleaned data from Phase 1
2. Splits content into optimal chunks
3. Creates embeddings using Sentence Transformers (CPU-friendly)
4. Stores in FAISS vector database (lighter than ChromaDB)
5. Tests retrieval accuracy
"""

import json
import os
import pickle
from datetime import datetime
from typing import List, Dict
import re
import warnings
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
        
        # CPU-friendly settings
        self.embedding_batch_size = 16
        
        # Statistics
        self.stats = {
            'total_pages': 0,
            'total_chunks': 0,
            'avg_chunk_size': 0,
            'embedding_dimension': 0
        }
    
    def load_cleaned_data(self):
        """Load cleaned data from Phase 1"""
        print(f"📂 Loading cleaned data from: {self.cleaned_data_file}")
        
        with open(self.cleaned_data_file, 'r', encoding='utf-8') as f:
            self.cleaned_data = json.load(f)
        
        self.stats['total_pages'] = len(self.cleaned_data)
        print(f"✅ Loaded {self.stats['total_pages']} cleaned pages\n")
    
    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """Split text into overlapping chunks"""
        chunks = []
        paragraphs = text.split('\n\n')
        
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
    
    def create_chunks(self):
        """Create chunks from all pages"""
        print("✂️  Creating chunks from cleaned pages...")
        print(f"   Chunk size: {self.chunk_size} chars")
        print(f"   Chunk overlap: {self.chunk_overlap} chars")
        print("=" * 60)
        
        for page in self.cleaned_data:
            page_id = page['page_id']
            title = page['title']
            url = page['url']
            content = page['content']
            
            print(f"\nPage {page_id}: {title[:50]}...")
            print(f"  Content length: {len(content)} chars")
            
            metadata = {
                'page_id': page_id,
                'title': title,
                'url': url,
                'source': 'website'
            }
            
            page_chunks = self.chunk_text(content, metadata)
            
            for i, chunk in enumerate(page_chunks):
                chunk['metadata']['chunk_id'] = f"{page_id}_{i+1}"
                chunk['metadata']['chunk_index'] = i + 1
                chunk['metadata']['total_chunks'] = len(page_chunks)
            
            self.chunks.extend(page_chunks)
            
            print(f"  Created: {len(page_chunks)} chunks")
        
        self.stats['total_chunks'] = len(self.chunks)
        chunk_sizes = [len(chunk['text']) for chunk in self.chunks]
        self.stats['avg_chunk_size'] = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        
        print("\n" + "=" * 60)
        print(f"✅ Total chunks created: {self.stats['total_chunks']}")
        print(f"   Average chunk size: {self.stats['avg_chunk_size']:.0f} chars")
        print()
    
    def initialize_embedding_model(self):
        """Initialize the sentence transformer model (CPU-optimized)"""
        print("🤖 Initializing embedding model (CPU mode)...")
        print("   Model: all-MiniLM-L6-v2")
        print("   Size: ~80MB (lightweight)")
        print("   Device: CPU")
        
        import torch
        device = 'cpu'
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.embedding_model.eval()
        
        self.stats['embedding_dimension'] = self.embedding_model.get_sentence_embedding_dimension()
        
        print(f"✅ Model loaded on CPU!")
        print(f"   Embedding dimension: {self.stats['embedding_dimension']}")
        print()
    
    def create_vector_database(self):
        """Create FAISS vector database"""
        print("🗄️  Creating FAISS vector database...")
        
        # Create output directory
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
        
        # Create FAISS index (CPU version, cosine similarity)
        self.index = faiss.IndexFlatIP(self.stats['embedding_dimension'])
        
        print(f"✅ Created FAISS index (dimension: {self.stats['embedding_dimension']})")
        print()
    
    def add_chunks_to_database(self):
        """Generate embeddings and add chunks to database"""
        print("📊 Generating embeddings and storing in database...")
        print(f"   Processing {len(self.chunks)} chunks...")
        print("=" * 60)
        
        import time
        
        # Extract all texts
        texts = [chunk['text'] for chunk in self.chunks]
        
        print(f"\n⚙️  Generating embeddings for all {len(texts)} chunks...")
        start_time = time.time()
        
        # Generate all embeddings
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.embedding_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # For cosine similarity
        )
        
        elapsed = time.time() - start_time
        speed = len(texts) / elapsed if elapsed > 0 else 0
        
        print(f"✅ Generated in {elapsed:.1f}s (~{speed:.1f} chunks/sec)")
        
        # Add to FAISS index
        print(f"\n💾 Adding embeddings to FAISS index...")
        self.index.add(embeddings)
        
        print(f"✅ All {self.index.ntotal} vectors stored!")
        print()
        
        # Save index and metadata
        print("💾 Saving database to disk...")
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(self.db_path, 'index.faiss'))
        
        # Save chunks metadata
        with open(os.path.join(self.db_path, 'chunks.pkl'), 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f"✅ Database saved to: {self.db_path}/")
        print()
    
    def test_retrieval(self, test_queries: List[str], n_results: int = 3):
        """Test the retrieval system"""
        print("🔍 Testing retrieval system...")
        print("=" * 60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Test Query {i}: '{query}'")
            print("-" * 60)
            
            # Encode query
            query_embedding = self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Search in FAISS
            distances, indices = self.index.search(query_embedding, n_results)
            
            # Display results
            for j, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    metadata = chunk['metadata']
                    text = chunk['text']
                    
                    similarity = float(dist)  # Already cosine similarity (0-1)
                    
                    print(f"\n  Result {j}:")
                    print(f"    Title: {metadata.get('title', 'N/A')[:60]}")
                    print(f"    URL: {metadata.get('url', 'N/A')}")
                    print(f"    Chunk: {metadata.get('chunk_index', 'N/A')}/{metadata.get('total_chunks', 'N/A')}")
                    print(f"    Relevance: {similarity:.3f} {'✅' if similarity > 0.7 else '⚠️' if similarity > 0.5 else '❌'}")
                    print(f"    Preview: {text[:150]}...")
        
        print("\n" + "=" * 60)
    
    def generate_report(self):
        """Generate Phase 2 report"""
        report = {
            'phase': 2,
            'timestamp': datetime.now().isoformat(),
            'input_file': self.cleaned_data_file,
            'statistics': self.stats,
            'configuration': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'embedding_model': 'all-MiniLM-L6-v2',
                'vector_db': 'FAISS',
                'db_path': self.db_path,
                'device': 'CPU'
            }
        }
        
        report_file = f'phase2_report_faiss_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print("\n" + "=" * 60)
        print("📊 PHASE 2 REPORT")
        print("=" * 60)
        print(f"\n📄 Data:")
        print(f"  Pages processed: {self.stats['total_pages']}")
        print(f"  Total chunks: {self.stats['total_chunks']}")
        print(f"  Avg chunk size: {self.stats['avg_chunk_size']:.0f} chars")
        print(f"\n🤖 Embeddings:")
        print(f"  Model: all-MiniLM-L6-v2")
        print(f"  Dimension: {self.stats['embedding_dimension']}")
        print(f"  Device: CPU")
        print(f"\n🗄️  Database:")
        print(f"  Type: FAISS (NO PROTOBUF!)")
        print(f"  Path: {self.db_path}/")
        print(f"  Stored vectors: {self.index.ntotal}")
        print(f"\n💾 Report saved to: {report_file}")
        print()


def main():
    """Main execution"""
    print("\n" + "=" * 60)
    print("🚀 PHASE 2: VECTOR DATABASE (FAISS - NO PROTOBUF)")
    print("=" * 60 + "\n")
    
    cleaned_dir = 'processed_data'
    
    if not os.path.exists(cleaned_dir):
        print(f"❌ Error: '{cleaned_dir}' directory not found!")
        print(f"💡 Please run Phase 1 first.")
        return
    
    json_files = [f for f in os.listdir(cleaned_dir) 
                  if f.startswith('clean_data') and f.endswith('.json')]
    
    if not json_files:
        print(f"❌ Error: No cleaned data files found!")
        return
    
    json_files.sort(reverse=True)
    cleaned_data_file = os.path.join(cleaned_dir, json_files[0])
    
    print(f"📂 Using cleaned data: {cleaned_data_file}\n")
    
    builder = VectorDatabaseBuilder(cleaned_data_file)
    
    builder.load_cleaned_data()
    builder.initialize_embedding_model()
    builder.create_chunks()
    builder.create_vector_database()
    builder.add_chunks_to_database()
    
    print("\n" + "=" * 60)
    print("🧪 TESTING RETRIEVAL SYSTEM")
    print("=" * 60)
    
    test_queries = [
        "What services does the company offer?",
        "How can I contact the company?",
        "Tell me about the company's technology",
        "What products are available?",
        "Company information and background"
    ]
    
    builder.test_retrieval(test_queries, n_results=3)
    builder.generate_report()
    
    print("=" * 60)
    print("✅ PHASE 2 COMPLETE!")
    print("=" * 60)
    print(f"\n📁 Vector database created at: {builder.db_path}/")
    print(f"📊 Total chunks indexed: {builder.stats['total_chunks']}")
    print(f"💻 Using: FAISS (No protobuf conflicts!)")
    print(f"\n🎯 Next: Run Phase 3 to integrate LLM")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()