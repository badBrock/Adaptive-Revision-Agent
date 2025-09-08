from pathlib import Path
import re
from typing import List
import difflib
import sys

class EnhancedKeywordFinder:
    def __init__(self, vault_path):
        self.vault_path = Path(vault_path)
        print(f"📁 Vault path: {vault_path}")
        
        # Load all files and extract common words for spell checking
        self.all_files = list(self.vault_path.rglob("*.md"))
        self.common_words = set()
        self._build_vocabulary()
        
    def _build_vocabulary(self):
        """Build vocabulary from all files for spell checking"""
        print("🔧 Building vocabulary for spell checking...")
        for file_path in self.all_files[:50]:  # Sample first 50 files for vocabulary
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    # Extract words (3+ characters)
                    words = re.findall(r'\b[a-zA-Z]{3,}\b', content)
                    self.common_words.update(words)
            except:
                continue
        print(f"✅ Built vocabulary with {len(self.common_words)} words")
    
    def spell_check_keyword(self, keyword):
        """Try to correct spelling of keyword"""
        if not self.common_words:
            return keyword
            
        # Get close matches
        suggestions = difflib.get_close_matches(
            keyword.lower(), 
            self.common_words, 
            n=3, 
            cutoff=0.6
        )
        
        if suggestions and suggestions[0] != keyword.lower():
            print(f"🔤 Did you mean: {', '.join(suggestions[:3])}?")
            return suggestions[0]
        
        return keyword
    
    def search_keyword(self, keyword, auto_correct=True):
        """Search for keyword with optional spell correction"""
        original_keyword = keyword
        print(f"\n🔍 Searching for keyword: '{keyword}'")
        
        if not self.all_files:
            print("❌ No .md files found in vault!")
            return []
        
        matching_files = []
        processed = 0
        
        for file_path in self.all_files:
            processed += 1
            if processed % 100 == 0:
                print(f"   Processed {processed}/{len(self.all_files)} files...")
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Case-insensitive substring search
                    if keyword.lower() in content.lower():
                        context = self.get_keyword_context(content, keyword)
                        matching_files.append({
                            'file_path': str(file_path),
                            'file_name': file_path.name,
                            'folder': str(file_path.parent),
                            'context': context,
                            'file_size': len(content)
                        })
                        
            except Exception as e:
                print(f"⚠️  Error reading {file_path}: {e}")
        
        # If no matches and auto-correct enabled, try spell correction
        if not matching_files and auto_correct:
            corrected_keyword = self.spell_check_keyword(keyword)
            if corrected_keyword != keyword.lower():
                print(f"🔄 Trying corrected spelling: '{corrected_keyword}'")
                return self.search_keyword(corrected_keyword, auto_correct=False)
        
        if matching_files:
            print(f"✅ Found {len(matching_files)} files containing '{keyword}'")
        else:
            print(f"❌ No files found containing '{keyword}'")
            
        return matching_files
    
    def get_keyword_context(self, content, keyword):
        """Get text around the keyword for preview"""
        lines = content.split('\n')
        context_lines = []
        
        for i, line in enumerate(lines):
            if keyword.lower() in line.lower():
                # Get surrounding lines for context
                start = max(0, i-1)
                end = min(len(lines), i+2)
                context = '\n'.join(lines[start:end])
                context_lines.append(context)
                
                if len(context_lines) >= 2:  # Limit context snippets
                    break
        
        return '\n...\n'.join(context_lines) if context_lines else "No context found"
    
    def display_search_results(self, matching_files):
        """Display search results with selection options"""
        if not matching_files:
            return
            
        print(f"\n📋 FOUND {len(matching_files)} FILES:")
        print("-" * 80)
        
        for i, file_info in enumerate(matching_files, 1):
            print(f"\n{i}. {file_info['file_name']}")
            print(f"   📂 Folder: {file_info['folder']}")
            print(f"   📏 Size: {file_info['file_size']:,} chars")
            print(f"   📝 Context Preview:")
            
            # Show context with limited lines
            context_lines = file_info['context'].split('\n')[:3]
            for line in context_lines:
                print(f"      {line[:100]}{'...' if len(line) > 100 else ''}")
    
    def select_multiple_files(self, matching_files):
        """Allow user to select multiple files"""
        if not matching_files:
            return []
        
        print(f"\n🎯 SELECT FILES (Multiple Selection Supported):")
        print("Examples:")
        print("  • Single: 1")
        print("  • Multiple: 1,3,5")
        print("  • Range: 1-3")
        print("  • All: all")
        print("  • Quit: quit")
        
        while True:
            choice = input(f"\nYour selection (1-{len(matching_files)}): ").strip()
            
            if not choice:
                print("❌ Please make a selection")
                continue
            
            # Check for quit command
            if choice.lower() in ['quit', 'exit', 'q', 'stop']:
                print("👋 Exiting program...")
                sys.exit(0)
            
            if choice.lower() == 'all':
                return matching_files
            
            try:
                selected_indices = []
                
                # Handle comma-separated values
                parts = choice.split(',')
                for part in parts:
                    part = part.strip()
                    
                    # Handle ranges (e.g., "1-3")
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        selected_indices.extend(range(start-1, end))
                    else:
                        selected_indices.append(int(part) - 1)
                
                # Filter valid indices
                valid_indices = [i for i in selected_indices if 0 <= i < len(matching_files)]
                
                if not valid_indices:
                    print("❌ No valid selections made")
                    continue
                
                selected_files = [matching_files[i] for i in valid_indices]
                print(f"✅ Selected {len(selected_files)} files")
                return selected_files
                
            except ValueError:
                print("❌ Invalid format. Use numbers, commas, or ranges (e.g., 1,3,5 or 1-3)")
                continue
    
    def view_selected_files(self, selected_files):
        """Display content of selected files with quit option"""
        if not selected_files:
            print("❌ No files selected")
            return True
        
        print(f"\n📚 VIEWING {len(selected_files)} SELECTED FILES:")
        print("💡 Type 'quit' at any prompt to exit the program")
        print("=" * 100)
        
        for i, file_info in enumerate(selected_files, 1):
            print(f"\n📖 FILE {i}/{len(selected_files)}: {file_info['file_name']}")
            print(f"📂 Path: {file_info['file_path']}")
            print("-" * 100)
            
            try:
                with open(file_info['file_path'], 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(content)
            except Exception as e:
                print(f"❌ Error reading file: {e}")
            
            print("-" * 100)
            
            # Check if this is the last file
            if i < len(selected_files):
                user_input = input(f"\nPress Enter to view next file ({i+1}/{len(selected_files)}) or type 'quit' to exit: ").strip().lower()
                if user_input in ['quit', 'exit', 'q', 'stop']:
                    print("👋 Exiting program...")
                    sys.exit(0)
            else:
                # Last file - give option to quit or continue
                user_input = input("\nPress Enter to continue or type 'quit' to exit: ").strip().lower()
                if user_input in ['quit', 'exit', 'q', 'stop']:
                    print("👋 Exiting program...")
                    sys.exit(0)
        
        print("\n✅ Finished viewing all selected files")
        return True
    
    def interactive_search(self):
        """Main interactive search interface"""
        while True:
            print("\n" + "="*80)
            print("🔍 ENHANCED KEYWORD FILE FINDER")
            print("✨ Features: Multi-select • View multiple files • Spell checking")
            print("💡 Type 'quit' at any time to exit")
            print("="*80)
            
            keyword = input("\n💭 Enter keyword to search: ").strip()
            
            if not keyword:
                print("❌ Please enter a keyword")
                continue
                
            # Check for quit command
            if keyword.lower() in ['quit', 'exit', 'stop', 'q']:
                print("👋 Goodbye!")
                break
            
            # Search for files
            matching_files = self.search_keyword(keyword)
            
            if not matching_files:
                continue
            
            # Display results
            self.display_search_results(matching_files)
            
            # Let user select multiple files (has built-in quit check)
            selected_files = self.select_multiple_files(matching_files)
            
            if not selected_files:
                continue
            
            # View selected files (has built-in quit check)
            self.view_selected_files(selected_files)

if __name__ == "__main__":
    print("🧠 Enhanced Multi-Select Keyword Finder")
    print("Perfect for large Joplin vaults with spell checking!")
    
    vault_path = input("\nEnter path to your vault: ").strip()
    
    if not vault_path:
        print("❌ Please provide a valid path")
        sys.exit(1)
    
    if not Path(vault_path).exists():
        print(f"❌ Path {vault_path} doesn't exist")
        sys.exit(1)
    
    try:
        finder = EnhancedKeywordFinder(vault_path)
        finder.interactive_search()
    except KeyboardInterrupt:
        print("\n👋 Program interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
    finally:
        print("Program ended.")
