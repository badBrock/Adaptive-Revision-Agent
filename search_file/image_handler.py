import re
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

class SimpleImageHandler:
    """Simple image handler for Obsidian attachments"""
    
    def __init__(self, vault_path: Path):
        self.vault_path = Path(vault_path)
        self.attachments_path = self.vault_path / "1 - Attachments"
        self.image_index = {}
        self.partial_name_index = {}
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.bmp'}
        self._build_image_index()
    
    def _build_image_index(self) -> None:
        """Build comprehensive index of all images"""
        print(f"🔍 Scanning for images in: {self.attachments_path}")
        
        if not self.attachments_path.exists():
            print(f"⚠️ Attachments folder not found: {self.attachments_path}")
            return
        
        image_count = 0
        for root, dirs, files in os.walk(self.attachments_path):
            for file in files:
                if Path(file).suffix.lower() in self.image_extensions:
                    full_path = Path(root) / file
                    self.image_index[file] = str(full_path)
                    
                    # Extract timestamp for partial matching (e.g., 20250905201943)
                    timestamp_match = re.search(r'(\d{14})', file)
                    if timestamp_match:
                        timestamp = timestamp_match.group(1)
                        self.partial_name_index[timestamp] = str(full_path)
                    
                    image_count += 1
        
        print(f"✅ Indexed {image_count} images")
    
    def find_image_path(self, filename: str) -> Optional[str]:
        """Find image path by exact filename or timestamp matching"""
        # First try exact match
        if filename in self.image_index:
            return self.image_index[filename]
        
        # Try timestamp matching (e.g., "20250905201943" from "Pasted image 20250905201943.png")
        timestamp_match = re.search(r'(\d{14})', filename)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            if timestamp in self.partial_name_index:
                return self.partial_name_index[timestamp]
        
        # Last resort: partial filename matching
        filename_lower = filename.lower()
        for indexed_filename, path in self.image_index.items():
            if filename_lower in indexed_filename.lower():
                return path
        
        return None
    
    def extract_image_references(self, content: str) -> List[str]:
        """Extract Obsidian image references from content"""
        pattern = r'!\[\[([^|\]]+\.(png|jpg|jpeg|gif|svg|webp|bmp))\]\]'
        matches = re.findall(pattern, content, re.IGNORECASE)
        return [match[0] for match in matches]
    
    def process_content_with_images(self, content: str) -> str:
        """Process content and convert Obsidian image links to standard markdown"""
        def replace_image_link(match):
            filename = match.group(1)
            image_path = self.find_image_path(filename)
            
            if not image_path:
                return f"![Image not found: {filename}]()"
            
            try:
                rel_path = os.path.relpath(image_path, self.vault_path)
                return f"![{filename}]({rel_path})"
            except:
                return f"![{filename}](file://{image_path})"
        
        pattern = r'!\[\[([^|\]]+\.(png|jpg|jpeg|gif|svg|webp|bmp))\]\]'
        processed_content = re.sub(pattern, replace_image_link, content, flags=re.IGNORECASE)
        return processed_content
    
    def get_image_info_for_file(self, file_path: str) -> List[Dict]:
        """Get image information for a specific file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            image_refs = self.extract_image_references(content)
            image_info = []
            
            for filename in image_refs:
                image_path = self.find_image_path(filename)
                image_info.append({
                    'filename': filename,
                    'path': image_path,
                    'exists': image_path is not None
                })
            
            return image_info
        except Exception:
            return []
    
    def copy_document_images(self, file_path: str, destination_folder: Path) -> int:
        """Copy all images from a specific document to destination folder"""
        destination_folder.mkdir(parents=True, exist_ok=True)
        
        image_info = self.get_image_info_for_file(file_path)
        copied_count = 0
        
        for img in image_info:
            if img['exists']:
                try:
                    source_path = Path(img['path'])
                    dest_path = destination_folder / source_path.name
                    shutil.copy2(source_path, dest_path)
                    print(f"📋 Copied: {img['filename']}")
                    copied_count += 1
                except Exception as e:
                    print(f"⚠️ Error copying {img['filename']}: {e}")
            else:
                print(f"❌ Image not found: {img['filename']}")
        
        return copied_count
    
    def process_content_for_export(self, content: str) -> str:
        """Process content for export - replace Obsidian links with local image references"""
        def replace_for_export(match):
            filename = match.group(1)
            # Replace with local images folder reference
            return f"![{filename}](images/{filename})"
        
        pattern = r'!\[\[([^|\]]+\.(png|jpg|jpeg|gif|svg|webp|bmp))\]\]'
        processed_content = re.sub(pattern, replace_for_export, content, flags=re.IGNORECASE)
        return processed_content
    
    def debug_search(self, filename: str) -> Dict:
        """Debug function to show search process"""
        result = {
            'filename': filename,
            'exact_match': filename in self.image_index,
            'partial_matches': [],
            'timestamp_found': None
        }
        
        timestamp_match = re.search(r'(\d{14})', filename)
        if timestamp_match:
            result['timestamp_found'] = timestamp_match.group(1)
            result['timestamp_in_index'] = timestamp_match.group(1) in self.partial_name_index
        
        filename_lower = filename.lower()
        for indexed_filename in self.image_index.keys():
            if filename_lower in indexed_filename.lower():
                result['partial_matches'].append(indexed_filename)
        
        return result
