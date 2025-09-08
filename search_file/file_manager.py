import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

class FileManager:
    """Handles all file I/O operations"""
    
    def __init__(self, index_file: Path, metadata_file: Path, hashes_file: Path):
        self.index_file = index_file
        self.metadata_file = metadata_file 
        self.hashes_file = hashes_file
    
    def load_file_hashes(self) -> Dict[str, str]:
        """Load file modification tracking"""
        if self.hashes_file.exists():
            try:
                with open(self.hashes_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading file hashes: {e}")
        
        return {}
    
    def save_file_hashes(self, file_hashes: Dict[str, str]) -> None:
        """Save file modification tracking"""
        try:
            with open(self.hashes_file, 'w') as f:
                json.dump(file_hashes, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving file hashes: {e}")
    
    def load_semantic_index(self) -> Any:
        """Load existing semantic index"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"⚠️ Error loading index: {e}")
        
        return None
    
    def save_semantic_index(self, semantic_index: Dict, vault_path: str) -> None:
        """Save semantic index to disk"""
        if semantic_index:
            try:
                with open(self.index_file, 'wb') as f:
                    pickle.dump(semantic_index, f)
                
                # Save metadata with vault path
                metadata = {
                    'total_chunks': semantic_index['total_chunks'],
                    'total_files': semantic_index['total_files'],
                    'last_updated': datetime.now().isoformat(),
                    'model_name': semantic_index['model_name'],
                    'vault_path': vault_path
                }
                
                with open(self.metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print("💾 Semantic index saved")
            except Exception as e:
                print(f"❌ Error saving index: {e}")
    
    def validate_vault_path(self, current_vault_path: str) -> bool:
        """Ensure the index matches the current vault path"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                stored_vault_path = metadata.get('vault_path', '')
                
                if stored_vault_path != current_vault_path:
                    print(f"⚠️ Vault path mismatch!")
                    print(f"   Stored: {stored_vault_path}")
                    print(f"   Current: {current_vault_path}")
                    print(f"🔄 Will rebuild index for new vault...")
                    return False
                    
            except Exception as e:
                print(f"⚠️ Error validating vault path: {e}")
                return False
        
        return True
