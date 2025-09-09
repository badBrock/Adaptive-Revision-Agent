from typing import Dict, Any, List, Optional
from pathlib import Path
import os
import re
import base64
import logging

logger = logging.getLogger(__name__)

class ContentCache:
    """Load .md files and images ONCE, cache in state to avoid re-sending"""
    
    @staticmethod
    def extract_image_paths(md_text: str, folder_path: str) -> List[str]:
        """Extract image paths from markdown text"""
        img_pattern = r'!\[.*?\]\((.*?)\)'
        relative_paths = re.findall(img_pattern, md_text)
        absolute_paths = []
        
        for img_path in relative_paths:
            if not os.path.isabs(img_path):
                abs_path = os.path.join(folder_path, img_path)
                if os.path.exists(abs_path):
                    absolute_paths.append(abs_path)
            else:
                if os.path.exists(img_path):
                    absolute_paths.append(img_path)
        
        return absolute_paths

    @staticmethod
    def encode_image_to_base64(image_path: str) -> Optional[str]:
        """Convert image to base64 for Groq API (cached once)"""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
            image_format = os.path.splitext(image_path)[1][1:].lower()
            if image_format == 'jpg':
                image_format = 'jpeg'
            
            return f"data:image/{image_format};base64,{encoded_string}"
        except Exception as e:
            logger.error(f"❌ Error encoding image {image_path}: {str(e)}")
            return None

    @staticmethod
    def load_content_cache(folder_path: str) -> Dict[str, Any]:
        """Load all content from folder and return cached data structure"""
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder path does not exist: {folder_path}")
        
        if not os.path.isdir(folder_path):
            raise ValueError(f"Path is not a directory: {folder_path}")

        # Discover markdown files
        documents = []
        all_image_paths = []
        
        md_files = list(Path(folder_path).glob("**/*.md"))
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    md_content = f.read()
                
                # Extract images referenced in markdown
                image_paths = ContentCache.extract_image_paths(md_content, folder_path)
                all_image_paths.extend(image_paths)
                
                documents.append({
                    "filename": md_file.name,
                    "filepath": str(md_file),
                    "markdown": md_content,
                    "word_count": len(md_content.split()),
                    "topic_name": md_file.stem
                })
                
            except Exception as e:
                logger.warning(f"Could not load {md_file}: {str(e)}")
        
        # Find ALL images in folder (not just referenced ones)
        for file_path in Path(folder_path).rglob("*"):
            if file_path.suffix.lower() in image_extensions:
                abs_path = str(file_path.absolute())
                if abs_path not in all_image_paths:
                    all_image_paths.append(abs_path)
        
        # Pre-encode ALL images to base64 (ONCE)
        encoded_images = []
        for img_path in all_image_paths:
            base64_img = ContentCache.encode_image_to_base64(img_path)
            if base64_img:
                encoded_images.append({
                    "path": img_path,
                    "filename": os.path.basename(img_path),
                    "base64": base64_img
                })
        
        content_cache = {
            "folder_path": os.path.abspath(folder_path),
            "documents": documents,
            "images": encoded_images,
            "total_documents": len(documents),
            "total_images": len(encoded_images),
            "cache_timestamp": None
        }
        
        logger.info(f"✅ Content cached: {len(documents)} documents, {len(encoded_images)} images")
        return content_cache

    @staticmethod
    def get_content_summary(cache: Dict[str, Any]) -> str:
        """Generate summary text for LLM context"""
        docs_summary = []
        for doc in cache["documents"]:
            docs_summary.append(f"- {doc['topic_name']}: {doc['word_count']} words")
        
        return f"""
Content Summary:
Documents ({cache['total_documents']}):
{chr(10).join(docs_summary)}
Images: {cache['total_images']} files
Folder: {cache['folder_path']}
"""

    @staticmethod
    def get_combined_text(cache: Dict[str, Any], max_chars: int = 3000) -> str:
        """Get combined text content from all documents, limited by max_chars"""
        combined = "\n\n".join([doc["markdown"] for doc in cache["documents"]])
        return combined[:max_chars] if len(combined) > max_chars else combined

    @staticmethod
    def get_base64_images(cache: Dict[str, Any], max_images: int = 3) -> List[str]:
        """Get list of base64 encoded images, limited by max_images"""
        return [img["base64"] for img in cache["images"][:max_images]]
