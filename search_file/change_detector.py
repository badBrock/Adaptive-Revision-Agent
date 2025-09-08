import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List

class ChangeDetector:
    """Handles change detection and file monitoring"""
    
    def __init__(self, vault_path: Path, vault_name: str):
        self.vault_path = vault_path
        self.vault_name = vault_name
        self.last_run_file = Path(f"last_run_{vault_name}.json")
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate file content hash"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return hashlib.md5(content.encode()).hexdigest()
        except Exception:
            return ""
    
    def load_last_run_info(self) -> Dict:
        """Load information about the last run"""
        if self.last_run_file.exists():
            try:
                with open(self.last_run_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading last run info: {e}")
        
        return {'last_shutdown': None, 'vault_path': str(self.vault_path)}

    def save_last_run_info(self, file_hashes: Dict) -> None:
        """Save information about this run"""
        last_run_info = {
            'last_shutdown': datetime.now().isoformat(),
            'vault_path': str(self.vault_path),
            'total_files_tracked': len(file_hashes)
        }
        
        try:
            with open(self.last_run_file, 'w') as f:
                json.dump(last_run_info, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving last run info: {e}")

    def detect_changes_since_last_run(self, file_hashes: Dict, update_queue) -> int:
        """Detect and queue files that changed while app was offline"""
        print("🔍 Scanning for changes since last run...")
        
        md_files = list(self.vault_path.rglob("*.md"))
        changes_found = 0
        new_files = 0
        
        for file_path in md_files:
            file_path_str = str(file_path)
            current_hash = self.calculate_file_hash(file_path_str)
            
            if not current_hash:
                continue
                
            stored_hash = file_hashes.get(file_path_str)
            
            if stored_hash is None:
                print(f"📄 New file: {file_path.name}")
                update_queue.put(file_path_str)
                new_files += 1
            elif current_hash != stored_hash:
                print(f"📝 Modified: {file_path.name}")
                update_queue.put(file_path_str)
                changes_found += 1
        
        # Check for deleted files
        deleted_files = []
        for stored_file_path in list(file_hashes.keys()):
            if not Path(stored_file_path).exists():
                print(f"🗑️ Deleted: {Path(stored_file_path).name}")
                deleted_files.append(stored_file_path)
        
        total_changes = changes_found + new_files + len(deleted_files)
        
        if total_changes > 0:
            print(f"📊 Found {total_changes} changes:")
            print(f"   📝 Modified: {changes_found}")
            print(f"   📄 New: {new_files}")
            print(f"   🗑️ Deleted: {len(deleted_files)}")
            print("🔄 Processing updates in background...")
        else:
            print("✅ No changes detected - vault is up to date!")
        
        return total_changes, deleted_files
