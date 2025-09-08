# from pathlib import Path
# import pickle
# import json
# import hashlib
# from datetime import datetime
# from typing import List, Dict, Set
# import threading
# import time
# import re
# from queue import Queue, Empty

# # Try to import AI components
# SEMANTIC_AVAILABLE = False
# try:
#     import numpy as np
#     from sentence_transformers import SentenceTransformer
#     SEMANTIC_AVAILABLE = True
# except ImportError:
#     pass

# # File monitoring
# try:
#     from watchdog.observers import Observer
#     from watchdog.events import FileSystemEventHandler
#     WATCHDOG_AVAILABLE = True
# except ImportError:
#     WATCHDOG_AVAILABLE = False
#     print("⚠️ Install watchdog for auto-updates: pip install watchdog")

# class DynamicVectorStore:
#     def __init__(self, vault_path: str):
#         self.vault_path = Path(vault_path)
        
#         # Create vault-specific filenames using vault path hash
#         vault_hash = hashlib.md5(str(self.vault_path).encode()).hexdigest()[:8]
#         vault_name = self.vault_path.name.replace(' ', '_').replace('\\', '_').replace('/', '_')
        
#         # Vault-specific index files
#         self.index_file = Path(f"vault_index_{vault_name}_{vault_hash}.pkl")
#         self.metadata_file = Path(f"vault_metadata_{vault_name}_{vault_hash}.json")
#         self.file_hashes_file = Path(f"vault_hashes_{vault_name}_{vault_hash}.json")
        
#         print(f"📁 Using vault-specific index: {self.index_file.name}")
        
#         # Core components
#         self.semantic_index = None
#         self.semantic_model = None
#         self.file_hashes = {}
#         self.update_queue = Queue()
#         self.is_updating = False
        
#         # Load last run information
#         self.last_run_info = self._load_last_run_info()
        
#         # Load existing data (now vault-specific)
#         self._load_file_hashes()
        
#         if SEMANTIC_AVAILABLE:
#             self._load_semantic_index()
#             self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            
#             # AUTO-BUILD: Check if we need to build initial index
#             self._check_and_build_initial_index()
            
#             # CHANGE DETECTION: Check for offline changes
#             if self.semantic_index is not None:
#                 self._detect_changes_since_last_run()
        
#         # Start background update worker
#         self.update_worker = threading.Thread(target=self._update_worker, daemon=True)
#         self.update_worker.start()
        
#         # Start file monitoring
#         if WATCHDOG_AVAILABLE:
#             self._start_file_monitoring()
    
    
#     def _check_and_build_initial_index(self):
#         """Check if we need to build initial index for new/empty vault"""
#         # If no semantic index exists, build one
#         if self.semantic_index is None:
#             md_files = list(self.vault_path.rglob("*.md"))
            
#             if md_files:
#                 print(f"🆕 New vault detected with {len(md_files)} files!")
#                 print("🔄 Building initial semantic index...")
                
#                 # Build index for all files
#                 self.force_full_rebuild()
#             else:
#                 print("📭 Empty vault - no markdown files found")
        
#         # Also check if existing index is empty but vault has files
#         elif (self.semantic_index.get('total_chunks', 0) == 0 or 
#               self.semantic_index.get('total_files', 0) == 0):
#             md_files = list(self.vault_path.rglob("*.md"))
            
#             if md_files:
#                 print(f"🔄 Found {len(md_files)} files but empty index - rebuilding...")
#                 self.force_full_rebuild()
    
#     def _validate_vault_path(self):
#         """Ensure the index matches the current vault path"""
#         if self.metadata_file.exists():
#             try:
#                 with open(self.metadata_file, 'r') as f:
#                     metadata = json.load(f)
                
#                 stored_vault_path = metadata.get('vault_path', '')
#                 current_vault_path = str(self.vault_path)
                
#                 if stored_vault_path != current_vault_path:
#                     print(f"⚠️ Vault path mismatch!")
#                     print(f"   Stored: {stored_vault_path}")
#                     print(f"   Current: {current_vault_path}")
#                     print(f"🔄 Will rebuild index for new vault...")
                    
#                     # Clear existing index for new vault
#                     self.semantic_index = None
#                     self.file_hashes = {}
#                     return False
                    
#             except Exception as e:
#                 print(f"⚠️ Error validating vault path: {e}")
#                 return False
        
#         return True
    
#     def _load_file_hashes(self):
#         """Load file modification tracking"""
#         if self.file_hashes_file.exists():
#             try:
#                 with open(self.file_hashes_file, 'r') as f:
#                     self.file_hashes = json.load(f)
#             except Exception as e:
#                 print(f"⚠️ Error loading file hashes: {e}")
#                 self.file_hashes = {}
#         else:
#             self.file_hashes = {}
    
#     def _save_file_hashes(self):
#         """Save file modification tracking"""
#         try:
#             with open(self.file_hashes_file, 'w') as f:
#                 json.dump(self.file_hashes, f, indent=2)
#         except Exception as e:
#             print(f"❌ Error saving file hashes: {e}")
    
#     def _calculate_file_hash(self, file_path: str) -> str:
#         """Calculate file content hash"""
#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 content = f.read()
#             return hashlib.md5(content.encode()).hexdigest()
#         except Exception:
#             return ""
    
#     def _load_semantic_index(self):
#         """Load existing semantic index"""
#         if self.index_file.exists():
#             try:
#                 # Validate vault path first
#                 if not self._validate_vault_path():
#                     print("🔄 Starting fresh index for new vault...")
#                     return
                
#                 with open(self.index_file, 'rb') as f:
#                     self.semantic_index = pickle.load(f)
#                 print(f"✅ Loaded existing semantic index ({self.semantic_index['total_chunks']} chunks)")
#             except Exception as e:
#                 print(f"⚠️ Error loading index: {e}")
#                 self.semantic_index = None
    
#     def _save_semantic_index(self):
#         """Save semantic index to disk"""
#         if self.semantic_index:
#             try:
#                 with open(self.index_file, 'wb') as f:
#                     pickle.dump(self.semantic_index, f)
                
#                 # Save metadata with vault path
#                 metadata = {
#                     'total_chunks': self.semantic_index['total_chunks'],
#                     'total_files': self.semantic_index['total_files'],
#                     'last_updated': datetime.now().isoformat(),
#                     'model_name': self.semantic_index['model_name'],
#                     'vault_path': str(self.vault_path)  # Store vault path for validation
#                 }
                
#                 with open(self.metadata_file, 'w') as f:
#                     json.dump(metadata, f, indent=2)
                
#                 print("💾 Semantic index saved")
#             except Exception as e:
#                 print(f"❌ Error saving index: {e}")
    
#     def should_chunk_file(self, file_path: str, content: str) -> bool:
#         """Decide whether to chunk a file based on its characteristics"""
        
#         # Very small files - keep whole
#         if len(content) < 1000:
#             return False
        
#         # Check file type patterns
#         file_name = Path(file_path).name.lower()
        
#         # Keep certain file types whole unless very large
#         if any(pattern in file_name for pattern in ['daily', 'journal', 'quick', 'note']):
#             return len(content) > 5000  # Only chunk if very large
        
#         return len(content) > 3000  # Default chunking threshold
    
#     def _chunk_text(self, text: str, max_chunk_size: int = 4000, min_chunk_size: int = 500) -> List[str]:
#         """Intelligent chunking for personal knowledge vaults"""
        
#         # If file is reasonably sized, keep it whole
#         if len(text) <= max_chunk_size:
#             return [text]
        
#         chunks = []
        
#         # Try to split by markdown headers first (better semantic boundaries)
#         if '##' in text or '###' in text:
#             chunks = self._split_by_headers(text, max_chunk_size)
        
#         # If header splitting didn't work well, try paragraph splitting
#         if not chunks or len(chunks) == 1:
#             chunks = self._split_by_paragraphs(text, max_chunk_size, min_chunk_size)
        
#         # Filter out very small chunks unless they're important
#         filtered_chunks = []
#         for chunk in chunks:
#             chunk = chunk.strip()
#             if len(chunk) >= min_chunk_size or self._is_important_small_chunk(chunk):
#                 filtered_chunks.append(chunk)
        
#         return filtered_chunks if filtered_chunks else [text]
    
#     def _split_by_headers(self, text: str, max_chunk_size: int) -> List[str]:
#         """Split by markdown headers for better semantic chunks"""
        
#         # Find header positions
#         header_pattern = r'^(#{1,6})\s+(.+)$'
#         lines = text.split('\n')
#         header_indices = []
        
#         for i, line in enumerate(lines):
#             if re.match(header_pattern, line):
#                 header_indices.append(i)
        
#         if len(header_indices) < 2:
#             return []
        
#         chunks = []
#         for i in range(len(header_indices)):
#             start_idx = header_indices[i]
#             end_idx = header_indices[i + 1] if i + 1 < len(header_indices) else len(lines)
            
#             section = '\n'.join(lines[start_idx:end_idx]).strip()
            
#             # If section is too large, split it further
#             if len(section) > max_chunk_size:
#                 sub_chunks = self._split_by_paragraphs(section, max_chunk_size, 200)
#                 chunks.extend(sub_chunks)
#             else:
#                 chunks.append(section)
        
#         return chunks
    
#     def _split_by_paragraphs(self, text: str, max_chunk_size: int, min_chunk_size: int) -> List[str]:
#         """Smarter paragraph-based splitting"""
        
#         # Split by double newlines but be more intelligent
#         paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
#         chunks = []
#         current_chunk = ""
        
#         for paragraph in paragraphs:
#             # Check if adding this paragraph would exceed limit
#             potential_chunk = current_chunk + ("\n\n" + paragraph if current_chunk else paragraph)
            
#             if len(potential_chunk) <= max_chunk_size:
#                 current_chunk = potential_chunk
#             else:
#                 # Save current chunk if it meets minimum size
#                 if current_chunk and len(current_chunk) >= min_chunk_size:
#                     chunks.append(current_chunk)
                
#                 # Start new chunk with current paragraph
#                 if len(paragraph) > max_chunk_size:
#                     # Split very long paragraphs by sentences
#                     sentence_chunks = self._split_by_sentences(paragraph, max_chunk_size)
#                     chunks.extend(sentence_chunks)
#                     current_chunk = ""
#                 else:
#                     current_chunk = paragraph
        
#         # Don't forget the last chunk
#         if current_chunk and len(current_chunk) >= min_chunk_size:
#             chunks.append(current_chunk)
        
#         return chunks
    
#     def _split_by_sentences(self, text: str, max_chunk_size: int) -> List[str]:
#         """Split long paragraphs by sentences"""
        
#         # Simple sentence splitting
#         sentences = re.split(r'(?<=[.!?])\s+', text)
        
#         chunks = []
#         current_chunk = ""
        
#         for sentence in sentences:
#             if len(current_chunk + sentence) <= max_chunk_size:
#                 current_chunk += sentence + " "
#             else:
#                 if current_chunk:
#                     chunks.append(current_chunk.strip())
#                 current_chunk = sentence + " "
        
#         if current_chunk:
#             chunks.append(current_chunk.strip())
        
#         return chunks
    
#     def _is_important_small_chunk(self, chunk: str) -> bool:
#         """Check if a small chunk contains important information"""
#         important_indicators = [
            
#             'http://', 'https://',  # URLs
#             '@', '#',  # Tags or mentions
#             '[[', ']]',  # Wiki links
#             'TODO:', 'NOTE:', 'IMPORTANT:',  # Special markers
#         ]
        
#         chunk_lower = chunk.lower()
#         return any(indicator in chunk_lower for indicator in important_indicators)
    
#     def cleanup_old_indexes(self):
#         """Clean up old vault index files"""
#         current_dir = Path(".")
#         old_files = []
        
#         # Find old index files
#         for pattern in ["vault_index_*.pkl", "vault_metadata_*.json", "vault_hashes_*.json"]:
#             for file_path in current_dir.glob(pattern):
#                 if (file_path != self.index_file and 
#                     file_path != self.metadata_file and 
#                     file_path != self.file_hashes_file):
#                     old_files.append(file_path)
        
#         if old_files:
#             print(f"🧹 Found {len(old_files)} old index files")
#             cleanup = input("Clean up old index files? (y/n): ").strip().lower()
#             if cleanup.startswith('y'):
#                 for file_path in old_files:
#                     try:
#                         file_path.unlink()
#                         print(f"🗑️ Deleted: {file_path.name}")
#                     except Exception as e:
#                         print(f"❌ Error deleting {file_path.name}: {e}")
    
#     def _generate_chunk_id(self, file_path: str, chunk_index: int) -> str:
#         """Generate unique ID for each chunk"""
#         return f"{hashlib.md5(file_path.encode()).hexdigest()}_{chunk_index}"
    
#     def _remove_file_from_index(self, file_path: str):
#         """Remove all chunks of a file from the index"""
#         if not self.semantic_index:
#             return
        
#         # Find chunks belonging to this file
#         chunks_to_remove = []
#         metadata_to_keep = []
#         embeddings_to_keep = []
#         chunks_to_keep = []
        
#         for i, metadata in enumerate(self.semantic_index['metadata']):
#             if metadata['file_path'] == file_path:
#                 chunks_to_remove.append(i)
#             else:
#                 metadata_to_keep.append(metadata)
#                 embeddings_to_keep.append(self.semantic_index['embeddings'][i])
#                 chunks_to_keep.append(self.semantic_index['chunks'][i])
        
#         if chunks_to_remove:
#             print(f"🗑️ Removing {len(chunks_to_remove)} chunks from {Path(file_path).name}")
            
#             # Update index
#             self.semantic_index['metadata'] = metadata_to_keep
#             self.semantic_index['embeddings'] = np.array(embeddings_to_keep) if embeddings_to_keep else np.array([])
#             self.semantic_index['chunks'] = chunks_to_keep
#             self.semantic_index['total_chunks'] = len(chunks_to_keep)
    
#     def _add_file_to_index(self, file_path: str):
#         """Add/update a file in the semantic index"""
#         if not SEMANTIC_AVAILABLE or not self.semantic_model:
#             return
        
#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 content = f.read()
#         except Exception as e:
#             print(f"❌ Error reading {file_path}: {e}")
#             return
        
#         if len(content.strip()) < 100:  # Skip very small files
#             return
        
#         # Smart chunking decision
#         if not self.should_chunk_file(file_path, content):
#             chunks = [content]  # Keep whole file
#             print(f"📄 Processing {Path(file_path).name} (keeping whole file)")
#         else:
#             # Create chunks using improved method
#             chunks = self._chunk_text(content)
#             if not chunks:
#                 return
#             print(f"📄 Processing {Path(file_path).name} ({len(chunks)} intelligent chunks)")
        
#         # Generate embeddings
#         try:
#             embeddings = self.semantic_model.encode(chunks, show_progress_bar=False)
#         except Exception as e:
#             print(f"❌ Error creating embeddings: {e}")
#             return
        
#         # Create metadata
#         new_metadata = []
#         for i, chunk in enumerate(chunks):
#             new_metadata.append({
#                 'file_path': str(file_path),
#                 'file_name': Path(file_path).name,
#                 'folder': str(Path(file_path).parent),
#                 'chunk_index': i,
#                 'file_size': len(content),
#                 'chunk_id': self._generate_chunk_id(str(file_path), i),
#                 'is_whole_file': len(chunks) == 1  # Track if this is a whole file
#             })
        
#         # Initialize index if it doesn't exist
#         if self.semantic_index is None:
#             self.semantic_index = {
#                 'embeddings': embeddings,
#                 'chunks': chunks,
#                 'metadata': new_metadata,
#                 'model_name': 'all-MiniLM-L6-v2',
#                 'created_at': datetime.now().isoformat(),
#                 'total_chunks': len(chunks),
#                 'total_files': 1
#             }
#         else:
#             # Append to existing index
#             existing_embeddings = self.semantic_index['embeddings']
#             if len(existing_embeddings) > 0:
#                 combined_embeddings = np.vstack([existing_embeddings, embeddings])
#             else:
#                 combined_embeddings = embeddings
            
#             self.semantic_index['embeddings'] = combined_embeddings
#             self.semantic_index['chunks'].extend(chunks)
#             self.semantic_index['metadata'].extend(new_metadata)
#             self.semantic_index['total_chunks'] = len(self.semantic_index['chunks'])
            
#             # Count unique files
#             unique_files = set(meta['file_path'] for meta in self.semantic_index['metadata'])
#             self.semantic_index['total_files'] = len(unique_files)
        
#         chunk_info = "whole file" if len(chunks) == 1 else f"{len(chunks)} chunks"
#         print(f"✅ Added {chunk_info} from {Path(file_path).name}")
    
#     def _update_worker(self):
#         """Background worker to process updates with startup progress"""
#         startup_updates = 0
#         startup_mode = True
        
#         while True:
#             try:
#                 file_path = self.update_queue.get(timeout=1)
#                 if file_path is None:  # Shutdown signal
#                     break
                
#                 self.is_updating = True
                
#                 if startup_mode:
#                     startup_updates += 1
#                     remaining = self.update_queue.qsize()
#                     print(f"🔄 Startup update {startup_updates} (queue: {remaining}): {Path(file_path).name}")
#                 else:
#                     print(f"🔄 Updating: {Path(file_path).name}")
                
#                 try:
#                     # Remove old version
#                     self._remove_file_from_index(file_path)
                    
#                     # Add new version
#                     self._add_file_to_index(file_path)
                    
#                     # Update file hash
#                     self.file_hashes[file_path] = self._calculate_file_hash(file_path)
                    
#                     # Save changes
#                     self._save_semantic_index()
#                     self._save_file_hashes()
                    
#                     if startup_mode:
#                         if self.update_queue.qsize() == 0:
#                             print(f"✅ Startup updates complete! Processed {startup_updates} files")
#                             startup_mode = False
#                     else:
#                         print(f"✅ Updated: {Path(file_path).name}")
                    
#                 except Exception as e:
#                     print(f"❌ Error updating {Path(file_path).name}: {e}")
#                 finally:
#                     self.is_updating = False
#                     self.update_queue.task_done()
                    
#             except Empty:
#                 # Queue timeout - this is normal, just continue
#                 if startup_mode and startup_updates > 0:
#                     startup_mode = False
#                     print(f"✅ Startup scan complete! No more changes detected.")
#                 continue
#             except Exception as e:
#                 print(f"❌ Critical update worker error: {e}")
#                 self.is_updating = False

    
#     def _start_file_monitoring(self):
#         """Start monitoring file system changes"""
#         class FileChangeHandler(FileSystemEventHandler):
#             def __init__(self, vector_store):
#                 self.vector_store = vector_store
            
#             def on_modified(self, event):
#                 if not event.is_directory and event.src_path.endswith('.md'):
#                     self.vector_store.queue_file_update(event.src_path)
            
#             def on_created(self, event):
#                 if not event.is_directory and event.src_path.endswith('.md'):
#                     self.vector_store.queue_file_update(event.src_path)
            
#             def on_deleted(self, event):
#                 if not event.is_directory and event.src_path.endswith('.md'):
#                     self.vector_store.queue_file_deletion(event.src_path)
        
#         self.observer = Observer()
#         event_handler = FileChangeHandler(self)
#         self.observer.schedule(event_handler, str(self.vault_path), recursive=True)
#         self.observer.start()
#         print(f"👀 Started monitoring {self.vault_path} for changes")
    
#     def queue_file_update(self, file_path: str):
#         """Queue a file for update"""
#         # Check if file actually changed
#         current_hash = self._calculate_file_hash(file_path)
#         if current_hash and current_hash != self.file_hashes.get(file_path):
#             print(f"📝 Detected change: {Path(file_path).name}")
#             self.update_queue.put(file_path)
    
#     def queue_file_deletion(self, file_path: str):
#         """Handle file deletion"""
#         if file_path in self.file_hashes:
#             print(f"🗑️ File deleted: {Path(file_path).name}")
#             self._remove_file_from_index(file_path)
#             del self.file_hashes[file_path]
#             self._save_semantic_index()
#             self._save_file_hashes()
    
#     def force_full_rebuild(self):
#         """Force a complete rebuild of the index"""
#         print("🔄 Starting full rebuild...")
        
#         # Clear existing index
#         self.semantic_index = None
#         self.file_hashes = {}
        
#         # Process all files
#         md_files = list(self.vault_path.rglob("*.md"))
#         print(f"📚 Rebuilding index for {len(md_files)} files...")
        
#         for file_path in md_files:
#             self._add_file_to_index(str(file_path))
#             self.file_hashes[str(file_path)] = self._calculate_file_hash(str(file_path))
        
#         self._save_semantic_index()
#         self._save_file_hashes()
        
#         # Show final stats
#         if self.semantic_index:
#             total_chunks = self.semantic_index['total_chunks']
#             total_files = self.semantic_index['total_files']
#             avg_chunks = total_chunks / total_files if total_files > 0 else 0
#             print(f"✅ Full rebuild complete!")
#             print(f"📊 {total_chunks} chunks from {total_files} files (avg: {avg_chunks:.1f} chunks/file)")
    
#     def get_status(self) -> Dict:
#         """Get current status of the vector store"""
#         status = {
#             'is_updating': self.is_updating,
#             'pending_updates': self.update_queue.qsize(),
#             'total_chunks': 0,
#             'total_files': 0,
#             'monitoring_active': hasattr(self, 'observer') and self.observer.is_alive(),
#             'vault_path': str(self.vault_path),
#             'index_file': str(self.index_file)
#         }
        
#         if self.semantic_index:
#             status['total_chunks'] = self.semantic_index['total_chunks']
#             status['total_files'] = self.semantic_index['total_files']
#             status['avg_chunks_per_file'] = status['total_chunks'] / status['total_files'] if status['total_files'] > 0 else 0
        
#         return status
    
#     def search(self, query: str, top_k: int = 10) -> List[Dict]:
#         """Search the dynamic vector store"""
#         if not SEMANTIC_AVAILABLE or not self.semantic_model or not self.semantic_index:
#             return []
        
#         try:
#             # Encode query
#             query_embedding = self.semantic_model.encode([query])
            
#             # Calculate similarities
#             embeddings = self.semantic_index['embeddings']
#             if len(embeddings) == 0:
#                 return []
                
#             similarities = np.dot(embeddings, query_embedding) / (
#                 np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
#             )
            
#             # Get top results
#             top_indices = np.argsort(similarities)[::-1][:top_k * 2]
            
#             # Group by file and deduplicate
#             file_results = {}
#             for idx in top_indices:
#                 metadata = self.semantic_index['metadata'][idx]
#                 file_path = metadata['file_path']
#                 similarity = similarities[idx]
#                 chunk_text = self.semantic_index['chunks'][idx]
                
#                 if file_path not in file_results:
#                     # Better context preview - show more for whole files
#                     context_length = 800 if metadata.get('is_whole_file', False) else 500
                    
#                     file_results[file_path] = {
#                         'file_path': file_path,
#                         'file_name': metadata['file_name'],
#                         'folder': metadata['folder'],
#                         'context': chunk_text[:context_length] + ("..." if len(chunk_text) > context_length else ""),
#                         'file_size': metadata['file_size'],
#                         'search_type': 'semantic',
#                         'relevance_score': similarity,
#                         'is_whole_file': metadata.get('is_whole_file', False)
#                     }
#                 elif similarity > file_results[file_path]['relevance_score']:
#                     context_length = 800 if metadata.get('is_whole_file', False) else 500
#                     file_results[file_path]['relevance_score'] = similarity
#                     file_results[file_path]['context'] = chunk_text[:context_length] + ("..." if len(chunk_text) > context_length else "")
#                     file_results[file_path]['is_whole_file'] = metadata.get('is_whole_file', False)
            
#             # Sort and return
#             results = list(file_results.values())
#             results.sort(key=lambda x: x['relevance_score'], reverse=True)
#             return results[:top_k]
            
#         except Exception as e:
#             print(f"❌ Search error: {e}")
#             return []
    
#     def shutdown(self):
#         """Clean shutdown with change tracking"""
#         print("💾 Saving shutdown state...")
        
#         # Save current file hashes and run info
#         self._save_file_hashes()
#         self._save_last_run_info()
        
#         if hasattr(self, 'observer'):
#             self.observer.stop()
#             self.observer.join()
        
#         # Signal update worker to stop
#         self.update_queue.put(None)
#         self.update_worker.join()
        
#         print("👋 Dynamic vector store shutdown complete")

#     def _load_last_run_info(self):
#         """Load information about the last run"""
#         last_run_file = Path(f"last_run_{self.vault_path.name.replace(' ', '_')}.json")
        
#         if last_run_file.exists():
#             try:
#                 with open(last_run_file, 'r') as f:
#                     return json.load(f)
#             except Exception as e:
#                 print(f"⚠️ Error loading last run info: {e}")
        
#         return {'last_shutdown': None, 'vault_path': str(self.vault_path)}

#     def _save_last_run_info(self):
#         """Save information about this run"""
#         last_run_file = Path(f"last_run_{self.vault_path.name.replace(' ', '_')}.json")
        
#         last_run_info = {
#             'last_shutdown': datetime.now().isoformat(),
#             'vault_path': str(self.vault_path),
#             'total_files_tracked': len(self.file_hashes)
#         }
        
#         try:
#             with open(last_run_file, 'w') as f:
#                 json.dump(last_run_info, f, indent=2)
#         except Exception as e:
#             print(f"❌ Error saving last run info: {e}")

#     def _detect_changes_since_last_run(self):
#         """Detect and queue files that changed while app was offline"""
#         print("🔍 Scanning for changes since last run...")
        
#         md_files = list(self.vault_path.rglob("*.md"))
#         changes_found = 0
#         new_files = 0
        
#         for file_path in md_files:
#             file_path_str = str(file_path)
#             current_hash = self._calculate_file_hash(file_path_str)
            
#             if not current_hash:  # Skip files that can't be read
#                 continue
                
#             stored_hash = self.file_hashes.get(file_path_str)
            
#             if stored_hash is None:
#                 # New file discovered
#                 print(f"📄 New file: {file_path.name}")
#                 self.update_queue.put(file_path_str)
#                 new_files += 1
#             elif current_hash != stored_hash:
#                 # File has been modified
#                 print(f"📝 Modified: {file_path.name}")
#                 self.update_queue.put(file_path_str)
#                 changes_found += 1
        
#         # Check for deleted files
#         deleted_files = []
#         for stored_file_path in list(self.file_hashes.keys()):
#             if not Path(stored_file_path).exists():
#                 print(f"🗑️ Deleted: {Path(stored_file_path).name}")
#                 self.queue_file_deletion(stored_file_path)
#                 deleted_files.append(stored_file_path)
        
#         total_changes = changes_found + new_files + len(deleted_files)
        
#         if total_changes > 0:
#             print(f"📊 Found {total_changes} changes:")
#             print(f"   📝 Modified: {changes_found}")
#             print(f"   📄 New: {new_files}")
#             print(f"   🗑️ Deleted: {len(deleted_files)}")
#             print("🔄 Processing updates in background...")
#         else:
#             print("✅ No changes detected - vault is up to date!")
        
#         return total_changes
from pathlib import Path
import hashlib
import threading
from queue import Queue, Empty
from typing import List, Dict
from datetime import datetime

# Import our modular components
from chunking_engine import ChunkingEngine
from change_detector import ChangeDetector
from file_manager import FileManager

# Try to import AI components
SEMANTIC_AVAILABLE = False
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    pass

# File monitoring
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("⚠️ Install watchdog for auto-updates: pip install watchdog")

class DynamicVectorStore:
    """Lightweight main coordinator class"""
    
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        
        # Create vault-specific filenames
        vault_hash = hashlib.md5(str(self.vault_path).encode()).hexdigest()[:8]
        vault_name = self.vault_path.name.replace(' ', '_').replace('\\', '_').replace('/', '_')
        
        # Initialize components
        self.chunking_engine = ChunkingEngine()
        self.change_detector = ChangeDetector(self.vault_path, vault_name)
        self.file_manager = FileManager(
            Path(f"vault_index_{vault_name}_{vault_hash}.pkl"),
            Path(f"vault_metadata_{vault_name}_{vault_hash}.json"),
            Path(f"vault_hashes_{vault_name}_{vault_hash}.json")
        )
        
        print(f"📁 Using vault-specific index: {self.file_manager.index_file.name}")
        
        # Core state
        self.semantic_index = None
        self.semantic_model = None
        self.file_hashes = {}
        self.update_queue = Queue()
        self.is_updating = False
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the vector store"""
        # Load existing data
        self.last_run_info = self.change_detector.load_last_run_info()
        self.file_hashes = self.file_manager.load_file_hashes()
        
        if SEMANTIC_AVAILABLE:
            # Validate and load index
            if self.file_manager.validate_vault_path(str(self.vault_path)):
                self.semantic_index = self.file_manager.load_semantic_index()
                if self.semantic_index:
                    print(f"✅ Loaded existing semantic index ({self.semantic_index['total_chunks']} chunks)")
            
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Auto-build and change detection
            self._check_and_build_initial_index()
            if self.semantic_index is not None:
                total_changes, deleted_files = self.change_detector.detect_changes_since_last_run(
                    self.file_hashes, self.update_queue
                )
                # Handle deleted files
                for deleted_file in deleted_files:
                    self.queue_file_deletion(deleted_file)
        
        # Start background processes
        self.update_worker = threading.Thread(target=self._update_worker, daemon=True)
        self.update_worker.start()
        
        if WATCHDOG_AVAILABLE:
            self._start_file_monitoring()
    
    def _check_and_build_initial_index(self):
        """Check if we need to build initial index"""
        if self.semantic_index is None:
            md_files = list(self.vault_path.rglob("*.md"))
            if md_files:
                print(f"🆕 New vault detected with {len(md_files)} files!")
                print("🔄 Building initial semantic index...")
                self.force_full_rebuild()
            else:
                print("📭 Empty vault - no markdown files found")
        elif (self.semantic_index.get('total_chunks', 0) == 0 or 
              self.semantic_index.get('total_files', 0) == 0):
            md_files = list(self.vault_path.rglob("*.md"))
            if md_files:
                print(f"🔄 Found {len(md_files)} files but empty index - rebuilding...")
                self.force_full_rebuild()

    def _add_file_to_index(self, file_path: str):
        """Add/update a file in the semantic index - delegated to chunking engine"""
        if not SEMANTIC_AVAILABLE or not self.semantic_model:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            return

        if len(content.strip()) < 100:
            return

        # Use chunking engine
        if not self.chunking_engine.should_chunk_file(file_path, content):
            chunks = [content]
            print(f"📄 Processing {Path(file_path).name} (keeping whole file)")
        else:
            chunks = self.chunking_engine.chunk_text(content)
            if not chunks:
                return
            print(f"📄 Processing {Path(file_path).name} ({len(chunks)} intelligent chunks)")

        # Generate embeddings and update index
        try:
            embeddings = self.semantic_model.encode(chunks, show_progress_bar=False)
        except Exception as e:
            print(f"❌ Error creating embeddings: {e}")
            return

        self._update_semantic_index(file_path, chunks, embeddings, content)

    def _update_semantic_index(self, file_path: str, chunks: List[str], embeddings, content: str):
        """Update the semantic index with new chunks"""
        # Create metadata
        new_metadata = []
        for i, chunk in enumerate(chunks):
            new_metadata.append({
                'file_path': str(file_path),
                'file_name': Path(file_path).name,
                'folder': str(Path(file_path).parent),
                'chunk_index': i,
                'file_size': len(content),
                'chunk_id': hashlib.md5(f"{file_path}_{i}".encode()).hexdigest(),
                'is_whole_file': len(chunks) == 1
            })

        # Initialize or update index
        if self.semantic_index is None:
            self.semantic_index = {
                'embeddings': embeddings,
                'chunks': chunks,
                'metadata': new_metadata,
                'model_name': 'all-MiniLM-L6-v2',
                'created_at': datetime.now().isoformat(),
                'total_chunks': len(chunks),
                'total_files': 1
            }
        else:
            # Append to existing index
            existing_embeddings = self.semantic_index['embeddings']
            if len(existing_embeddings) > 0:
                combined_embeddings = np.vstack([existing_embeddings, embeddings])
            else:
                combined_embeddings = embeddings

            self.semantic_index['embeddings'] = combined_embeddings
            self.semantic_index['chunks'].extend(chunks)
            self.semantic_index['metadata'].extend(new_metadata)
            self.semantic_index['total_chunks'] = len(self.semantic_index['chunks'])

            unique_files = set(meta['file_path'] for meta in self.semantic_index['metadata'])
            self.semantic_index['total_files'] = len(unique_files)

        chunk_info = "whole file" if len(chunks) == 1 else f"{len(chunks)} chunks"
        print(f"✅ Added {chunk_info} from {Path(file_path).name}")

    def _remove_file_from_index(self, file_path: str):
        """Remove all chunks of a file from the index"""
        if not self.semantic_index:
            return
        
        # Find chunks belonging to this file
        chunks_to_remove = []
        metadata_to_keep = []
        embeddings_to_keep = []
        chunks_to_keep = []
        
        for i, metadata in enumerate(self.semantic_index['metadata']):
            if metadata['file_path'] == file_path:
                chunks_to_remove.append(i)
            else:
                metadata_to_keep.append(metadata)
                embeddings_to_keep.append(self.semantic_index['embeddings'][i])
                chunks_to_keep.append(self.semantic_index['chunks'][i])
        
        if chunks_to_remove:
            print(f"🗑️ Removing {len(chunks_to_remove)} chunks from {Path(file_path).name}")
            
            # Update index
            self.semantic_index['metadata'] = metadata_to_keep
            self.semantic_index['embeddings'] = np.array(embeddings_to_keep) if embeddings_to_keep else np.array([])
            self.semantic_index['chunks'] = chunks_to_keep
            self.semantic_index['total_chunks'] = len(chunks_to_keep)

    def _update_worker(self):
        """Background worker to process updates with startup progress"""
        startup_updates = 0
        startup_mode = True
        
        while True:
            try:
                file_path = self.update_queue.get(timeout=1)
                if file_path is None:  # Shutdown signal
                    break
                
                self.is_updating = True
                
                if startup_mode:
                    startup_updates += 1
                    remaining = self.update_queue.qsize()
                    print(f"🔄 Startup update {startup_updates} (queue: {remaining}): {Path(file_path).name}")
                else:
                    print(f"🔄 Updating: {Path(file_path).name}")
                
                try:
                    # Remove old version
                    self._remove_file_from_index(file_path)
                    
                    # Add new version
                    self._add_file_to_index(file_path)
                    
                    # Update file hash
                    self.file_hashes[file_path] = self.change_detector.calculate_file_hash(file_path)
                    
                    # Save changes
                    self.file_manager.save_semantic_index(self.semantic_index, str(self.vault_path))
                    self.file_manager.save_file_hashes(self.file_hashes)
                    
                    if startup_mode:
                        if self.update_queue.qsize() == 0:
                            print(f"✅ Startup updates complete! Processed {startup_updates} files")
                            startup_mode = False
                    else:
                        print(f"✅ Updated: {Path(file_path).name}")
                    
                except Exception as e:
                    print(f"❌ Error updating {Path(file_path).name}: {e}")
                finally:
                    self.is_updating = False
                    self.update_queue.task_done()
                    
            except Empty:
                # Queue timeout - this is normal, just continue
                if startup_mode and startup_updates > 0:
                    startup_mode = False
                    print(f"✅ Startup scan complete! No more changes detected.")
                continue
            except Exception as e:
                print(f"❌ Critical update worker error: {e}")
                self.is_updating = False

    def _start_file_monitoring(self):
        """Start monitoring file system changes"""
        class FileChangeHandler(FileSystemEventHandler):
            def __init__(self, vector_store):
                self.vector_store = vector_store
            
            def on_modified(self, event):
                if not event.is_directory and event.src_path.endswith('.md'):
                    self.vector_store.queue_file_update(event.src_path)
            
            def on_created(self, event):
                if not event.is_directory and event.src_path.endswith('.md'):
                    self.vector_store.queue_file_update(event.src_path)
            
            def on_deleted(self, event):
                if not event.is_directory and event.src_path.endswith('.md'):
                    self.vector_store.queue_file_deletion(event.src_path)
        
        self.observer = Observer()
        event_handler = FileChangeHandler(self)
        self.observer.schedule(event_handler, str(self.vault_path), recursive=True)
        self.observer.start()
        print(f"👀 Started monitoring {self.vault_path} for changes")

    def queue_file_update(self, file_path: str):
        """Queue a file for update"""
        # Check if file actually changed
        current_hash = self.change_detector.calculate_file_hash(file_path)
        if current_hash and current_hash != self.file_hashes.get(file_path):
            print(f"📝 Detected change: {Path(file_path).name}")
            self.update_queue.put(file_path)

    def queue_file_deletion(self, file_path: str):
        """Handle file deletion"""
        if file_path in self.file_hashes:
            print(f"🗑️ File deleted: {Path(file_path).name}")
            self._remove_file_from_index(file_path)
            del self.file_hashes[file_path]
            self.file_manager.save_semantic_index(self.semantic_index, str(self.vault_path))
            self.file_manager.save_file_hashes(self.file_hashes)

    def get_status(self) -> Dict:
        """Get current status of the vector store"""
        status = {
            'is_updating': self.is_updating,
            'pending_updates': self.update_queue.qsize(),
            'total_chunks': 0,
            'total_files': 0,
            'monitoring_active': hasattr(self, 'observer') and self.observer.is_alive(),
            'vault_path': str(self.vault_path),
            'index_file': str(self.file_manager.index_file)
        }
        
        if self.semantic_index:
            status['total_chunks'] = self.semantic_index['total_chunks']
            status['total_files'] = self.semantic_index['total_files']
            status['avg_chunks_per_file'] = status['total_chunks'] / status['total_files'] if status['total_files'] > 0 else 0
        
        return status

    def cleanup_old_indexes(self):
        """Clean up old vault index files"""
        current_dir = Path(".")
        old_files = []
        
        # Find old index files
        for pattern in ["vault_index_*.pkl", "vault_metadata_*.json", "vault_hashes_*.json"]:
            for file_path in current_dir.glob(pattern):
                if (file_path != self.file_manager.index_file and 
                    file_path != self.file_manager.metadata_file and 
                    file_path != self.file_manager.hashes_file):
                    old_files.append(file_path)
        
        if old_files:
            print(f"🧹 Found {len(old_files)} old index files")
            cleanup = input("Clean up old index files? (y/n): ").strip().lower()
            if cleanup.startswith('y'):
                for file_path in old_files:
                    try:
                        file_path.unlink()
                        print(f"🗑️ Deleted: {file_path.name}")
                    except Exception as e:
                        print(f"❌ Error deleting {file_path.name}: {e}")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search the dynamic vector store"""
        if not SEMANTIC_AVAILABLE or not self.semantic_model or not self.semantic_index:
            return []

        try:
            query_embedding = self.semantic_model.encode([query])[0]
            embeddings = self.semantic_index['embeddings']
            
            if len(embeddings) == 0:
                return []

            similarities = np.dot(embeddings, query_embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
            )

            # Get and process top results
            top_indices = np.argsort(similarities)[::-1][:top_k * 2]
            
            file_results = {}
            for idx in top_indices:
                metadata = self.semantic_index['metadata'][idx]
                file_path = metadata['file_path']
                similarity = similarities[idx]
                chunk_text = self.semantic_index['chunks'][idx]

                if file_path not in file_results:
                    context_length = 800 if metadata.get('is_whole_file', False) else 500
                    
                    file_results[file_path] = {
                        'file_path': file_path,
                        'file_name': metadata['file_name'],
                        'folder': metadata['folder'],
                        'context': chunk_text[:context_length] + ("..." if len(chunk_text) > context_length else ""),
                        'file_size': metadata['file_size'],
                        'search_type': 'semantic',
                        'relevance_score': similarity,
                        'is_whole_file': metadata.get('is_whole_file', False)
                    }
                elif similarity > file_results[file_path]['relevance_score']:
                    context_length = 800 if metadata.get('is_whole_file', False) else 500
                    file_results[file_path]['relevance_score'] = similarity
                    file_results[file_path]['context'] = chunk_text[:context_length] + ("..." if len(chunk_text) > context_length else "")

            results = list(file_results.values())
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return results[:top_k]

        except Exception as e:
            print(f"❌ Search error: {e}")
            return []

    def force_full_rebuild(self):
        """Force a complete rebuild of the index"""
        print("🔄 Starting full rebuild...")
        
        self.semantic_index = None
        self.file_hashes = {}
        
        md_files = list(self.vault_path.rglob("*.md"))
        print(f"📚 Rebuilding index for {len(md_files)} files...")
        
        for file_path in md_files:
            self._add_file_to_index(str(file_path))
            self.file_hashes[str(file_path)] = self.change_detector.calculate_file_hash(str(file_path))
        
        self.file_manager.save_semantic_index(self.semantic_index, str(self.vault_path))
        self.file_manager.save_file_hashes(self.file_hashes)
        
        if self.semantic_index:
            total_chunks = self.semantic_index['total_chunks']
            total_files = self.semantic_index['total_files']
            avg_chunks = total_chunks / total_files if total_files > 0 else 0
            print(f"✅ Full rebuild complete!")
            print(f"📊 {total_chunks} chunks from {total_files} files (avg: {avg_chunks:.1f} chunks/file)")

    def shutdown(self):
        """Clean shutdown"""
        print("💾 Saving shutdown state...")
        
        self.file_manager.save_file_hashes(self.file_hashes)
        self.change_detector.save_last_run_info(self.file_hashes)
        
        if hasattr(self, 'observer'):
            self.observer.stop()
            self.observer.join()
        
        self.update_queue.put(None)
        self.update_worker.join()
        
        print("👋 Dynamic vector store shutdown complete")
