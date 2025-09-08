from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import pickle
from datetime import datetime
from vault_config import VaultConfig

class VaultIndexer:
    def __init__(self):
        self.config = VaultConfig()
        self.vault_path = Path(self.config.get_vault_path())
        self.index_file = Path("vault_semantic_index.pkl")
        self.metadata_file = Path("vault_index_metadata.json")
        
    def chunk_text(self, text: str, max_chunk_size: int = 2000) -> list:
        """Smart chunking by paragraphs"""
        if len(text) <= max_chunk_size:
            return [text]
        
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def build_semantic_index(self):
        """Build and save semantic index - run this once"""
        print("🤖 BUILDING SEMANTIC INDEX")
        print("=" * 50)
        
        # Check if index already exists
        if self.index_file.exists():
            rebuild = input("Index already exists. Rebuild? (y/n): ").strip().lower()
            if not rebuild.startswith('y'):
                print("✅ Using existing index")
                return
        
        # Load AI model
        print("🧠 Loading AI model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get all markdown files
        md_files = list(self.vault_path.rglob("*.md"))
        print(f"📚 Processing {len(md_files)} markdown files...")
        
        # Process files and create index
        all_chunks = []
        chunk_metadata = []
        processed_files = 0
        
        for file_path in md_files:
            processed_files += 1
            if processed_files % 50 == 0:
                print(f"   Processed {processed_files}/{len(md_files)} files...")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if len(content.strip()) < 100:  # Skip very small files
                    continue
                
                # Create chunks
                chunks = self.chunk_text(content)
                
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) > 50:  # Skip tiny chunks
                        all_chunks.append(chunk)
                        chunk_metadata.append({
                            'file_path': str(file_path),
                            'file_name': file_path.name,
                            'folder': str(file_path.parent),
                            'chunk_index': i,
                            'file_size': len(content)
                        })
                        
            except Exception as e:
                print(f"⚠️  Error processing {file_path}: {e}")
        
        if not all_chunks:
            print("❌ No content found to index")
            return
        
        print(f"🔥 Creating embeddings for {len(all_chunks)} chunks...")
        
        # Create embeddings in batches to avoid memory issues
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            print(f"   Processing batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")
            
            batch_embeddings = model.encode(batch, show_progress_bar=True)
            all_embeddings.extend(batch_embeddings)
        
        # Create index structure
        index_data = {
            'embeddings': np.array(all_embeddings),
            'chunks': all_chunks,
            'metadata': chunk_metadata,
            'model_name': 'all-MiniLM-L6-v2',
            'created_at': datetime.now().isoformat(),
            'total_chunks': len(all_chunks),
            'total_files': len(set(meta['file_path'] for meta in chunk_metadata))
        }
        
        # Save index
        print("💾 Saving semantic index...")
        with open(self.index_file, 'wb') as f:
            pickle.dump(index_data, f)
        
        # Save metadata
        metadata = {
            'total_chunks': len(all_chunks),
            'total_files': index_data['total_files'],
            'created_at': index_data['created_at'],
            'model_name': index_data['model_name'],
            'vault_path': str(self.vault_path)
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Semantic index built successfully!")
        print(f"   📊 {len(all_chunks)} chunks from {index_data['total_files']} files")
        print(f"   💾 Saved to: {self.index_file}")
        
    def check_index_status(self):
        """Check if index exists and show info"""
        if not self.index_file.exists():
            print("❌ No semantic index found. Run build_semantic_index() first.")
            return False
        
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                
            print("✅ Semantic Index Status:")
            print(f"   📊 {metadata['total_chunks']} chunks from {metadata['total_files']} files")
            print(f"   📅 Created: {metadata['created_at']}")
            print(f"   🤖 Model: {metadata['model_name']}")
            return True
        
        return True

if __name__ == "__main__":
    print("🔧 VAULT SEMANTIC INDEXER")
    print("Build once, search fast!")
    print("=" * 40)
    
    indexer = VaultIndexer()
    
    while True:
        print("\n1. Check index status")
        print("2. Build/rebuild semantic index")
        print("3. Exit")
        
        choice = input("\nChoose option (1-3): ").strip()
        
        if choice == '1':
            indexer.check_index_status()
        elif choice == '2':
            indexer.build_semantic_index()
        elif choice == '3':
            print("👋 Done!")
            break
        else:
            print("❌ Invalid choice")
