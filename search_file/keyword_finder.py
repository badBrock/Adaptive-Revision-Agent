from pathlib import Path
import re
import difflib
from typing import List, Dict
from vault_config import VaultConfig

class FastKeywordFinder:
    def __init__(self):
        self.config = VaultConfig()
        self.vault_path = Path(self.config.get_vault_path())
        print(f"📁 Loading vault: {self.vault_path}")
        self.file_contents = {}
        self.common_words = set()
        self._load_all_files()
        self._build_vocabulary()

    def _load_all_files(self):
        """Load all files once"""
        md_files = list(self.vault_path.rglob("*.md"))
        print(f"📥 Loading {len(md_files)} files for keyword search...")
        
        for file_path in md_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if len(content.strip()) > 50:
                    self.file_contents[str(file_path)] = content
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")
        
        print(f"✅ Loaded {len(self.file_contents)} files")

    def _build_vocabulary(self):
        """Build spell-check vocabulary"""
        sample_files = list(self.file_contents.items())[:30]
        for _, content in sample_files:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
            self.common_words.update(words)

    def _should_spell_check(self, keyword: str) -> bool:
        """Decide if we should spell check this keyword"""
        # Don't spell check if:
        # 1. Contains numbers or special chars (likely proper nouns, codes, etc.)
        if re.search(r'[0-9\-_.]', keyword):
            return False
        
        # 2. Contains capital letters (likely proper nouns)
        if any(c.isupper() for c in keyword):
            return False
        
        # 3. Multiple words (likely names or phrases)
        if len(keyword.split()) > 1:
            return False
        
        # 4. Very short or very long
        if len(keyword) < 3 or len(keyword) > 15:
            return False
        
        # 5. Already a common word
        if keyword.lower() in self.common_words:
            return False
        
        return True

    def search(self, keyword: str, spell_check: bool = True) -> List[Dict]:
        """Fast keyword search with smarter spell checking"""
        print(f"🔍 Keyword search: '{keyword}'")
        
        results = []
        keyword_lower = keyword.lower()
        
        # Direct search first
        for file_path, content in self.file_contents.items():
            if keyword_lower in content.lower():
                context = self._get_context(content, keyword)
                results.append({
                    'file_path': file_path,
                    'file_name': Path(file_path).name,
                    'folder': str(Path(file_path).parent),
                    'context': context,
                    'file_size': len(content),
                    'search_type': 'keyword'
                })

        # Try spell correction ONLY if no results and keyword should be spell-checked
        if not results and spell_check and self._should_spell_check(keyword):
            corrected = self._spell_check(keyword)
            if corrected != keyword.lower():
                print(f"🔄 Trying spell correction: '{corrected}'")
                return self.search(corrected, spell_check=False)

        return results

    def _get_context(self, content: str, keyword: str) -> str:
        """Get context around keyword"""
        lines = content.split('\n')
        context_lines = []
        
        for i, line in enumerate(lines):
            if keyword.lower() in line.lower():
                start = max(0, i-1)
                end = min(len(lines), i+2)
                context = '\n'.join(lines[start:end])
                context_lines.append(context)
                if len(context_lines) >= 2:
                    break
        
        return '\n...\n'.join(context_lines) if context_lines else "No context"

    def _spell_check(self, keyword: str) -> str:
        """Conservative spell checking"""
        suggestions = difflib.get_close_matches(
            keyword.lower(), self.common_words, n=1, cutoff=0.8  # Higher cutoff = more conservative
        )
        return suggestions[0] if suggestions else keyword

if __name__ == "__main__":
    finder = FastKeywordFinder()
    
    while True:
        keyword = input("\n🔍 Enter keyword (or 'quit'): ").strip()
        if keyword.lower() in ['quit', 'exit']:
            break
            
        results = finder.search(keyword)
        print(f"Found {len(results)} files")
        
        for i, result in enumerate(results[:5], 1):
            print(f"{i}. {result['file_name']}")
