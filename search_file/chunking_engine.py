import re
from pathlib import Path
from typing import List

class ChunkingEngine:
    """Handles all text chunking logic"""
    
    def should_chunk_file(self, file_path: str, content: str) -> bool:
        """Decide whether to chunk a file based on its characteristics"""
        if len(content) < 1000:
            return False
        
        file_name = Path(file_path).name.lower()
        if any(pattern in file_name for pattern in ['daily', 'journal', 'quick', 'note']):
            return len(content) > 5000
        
        return len(content) > 3000

    def chunk_text(self, text: str, max_chunk_size: int = 4000, min_chunk_size: int = 500) -> List[str]:
        """Intelligent chunking for personal knowledge vaults"""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        
        # Try to split by markdown headers first
        if '##' in text or '###' in text:
            chunks = self._split_by_headers(text, max_chunk_size)
        
        if not chunks or len(chunks) == 1:
            chunks = self._split_by_paragraphs(text, max_chunk_size, min_chunk_size)
        
        # Filter out very small chunks unless they're important
        filtered_chunks = []
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) >= min_chunk_size or self._is_important_small_chunk(chunk):
                filtered_chunks.append(chunk)
        
        return filtered_chunks if filtered_chunks else [text]

    def _split_by_headers(self, text: str, max_chunk_size: int) -> List[str]:
        """Split by markdown headers for better semantic chunks"""
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = text.split('\n')
        header_indices = []
        
        for i, line in enumerate(lines):
            if re.match(header_pattern, line):
                header_indices.append(i)
        
        if len(header_indices) < 2:
            return []
        
        chunks = []
        for i in range(len(header_indices)):
            start_idx = header_indices[i]
            end_idx = header_indices[i + 1] if i + 1 < len(header_indices) else len(lines)
            section = '\n'.join(lines[start_idx:end_idx]).strip()
            
            if len(section) > max_chunk_size:
                sub_chunks = self._split_by_paragraphs(section, max_chunk_size, 200)
                chunks.extend(sub_chunks)
            else:
                chunks.append(section)
        
        return chunks

    def _split_by_paragraphs(self, text: str, max_chunk_size: int, min_chunk_size: int) -> List[str]:
        """Smarter paragraph-based splitting"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            potential_chunk = current_chunk + ("\n\n" + paragraph if current_chunk else paragraph)
            
            if len(potential_chunk) <= max_chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk and len(current_chunk) >= min_chunk_size:
                    chunks.append(current_chunk)
                
                if len(paragraph) > max_chunk_size:
                    sentence_chunks = self._split_by_sentences(paragraph, max_chunk_size)
                    chunks.extend(sentence_chunks)
                    current_chunk = ""
                else:
                    current_chunk = paragraph
        
        if current_chunk and len(current_chunk) >= min_chunk_size:
            chunks.append(current_chunk)
        
        return chunks

    def _split_by_sentences(self, text: str, max_chunk_size: int) -> List[str]:
        """Split long paragraphs by sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def _is_important_small_chunk(self, chunk: str) -> bool:
        """Check if a small chunk contains important information"""
        important_indicators = [
            
            'TODO:', 'NOTE:', 'IMPORTANT:'
        ]
        
        chunk_lower = chunk.lower()
        return any(indicator in chunk_lower for indicator in important_indicators)
