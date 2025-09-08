# from pathlib import Path
# import sys
# from typing import List, Dict
# import time

# # Import components
# try:
#     from keyword_finder import FastKeywordFinder
#     from vault_config import VaultConfig
#     from dynamic_vector_store import DynamicVectorStore
#     COMPONENTS_AVAILABLE = True
# except ImportError:
#     COMPONENTS_AVAILABLE = False
#     print("❌ Missing components")

# class EnhancedUnifiedFinder:
#     def __init__(self):
#         if not COMPONENTS_AVAILABLE:
#             print("❌ Cannot initialize. Missing required components.")
#             sys.exit(1)
            
#         # Initialize components
#         print("⚡ Initializing enhanced document finder...")
#         self.keyword_finder = FastKeywordFinder()
#         self.vault_path = Path(self.keyword_finder.config.get_vault_path())
        
#         # Initialize dynamic vector store
#         print("🔄 Initializing dynamic vector store...")
#         self.dynamic_vector_store = DynamicVectorStore(str(self.vault_path))
        
#         # Wait a moment for initial setup
#         time.sleep(1)
    
#     def smart_search(self, query: str) -> List[Dict]:
#         """Smart search with dynamic updates"""
#         query = query.strip()
#         word_count = len(query.split())
        
#         # Show update status
#         status = self.dynamic_vector_store.get_status()
#         if status['is_updating']:
#             print(f"🔄 (Updating in background - {status['pending_updates']} files pending)")
        
#         if word_count <= 3:
#             print(f"⚡ Using KEYWORD search ({word_count} words)")
#             return self.keyword_finder.search(query)
#         else:
#             print(f"🧠 Using SEMANTIC search ({word_count} words)")
#             results = self.dynamic_vector_store.search(query)
            
#             if not results:
#                 print("🔄 No semantic results, trying keyword search...")
#                 key_terms = [w for w in query.split() if len(w) > 3][:2]
#                 if key_terms:
#                     return self.keyword_finder.search(" ".join(key_terms))
            
#             return results
    
#     def display_results(self, results: List[Dict]):
#         """Display search results"""
#         if not results:
#             print("❌ No results found")
#             return
        
#         print(f"\n📋 FOUND {len(results)} DOCUMENTS:")
#         print("=" * 80)
        
#         for i, result in enumerate(results, 1):
#             print(f"\n{i}. {result['file_name']}")
#             print(f"   📂 {result['folder']}")
#             print(f"   🔍 {result['search_type'].upper()}")
            
#             if result.get('relevance_score'):
#                 print(f"   📈 Relevance: {result['relevance_score']:.3f}")
            
#             print(f"   📝 Preview:")
#             context_lines = result['context'].split('\n')[:2]
#             for line in context_lines:
#                 print(f"      {line[:100]}{'...' if len(line) > 100 else ''}")
    
#     def select_and_view(self, results: List[Dict]):
#         """Handle file selection and viewing"""
#         if not results:
#             return
        
#         choice = input(f"\nSelect files (1-{len(results)}, ranges, 'all', 'status', 'quit'): ").strip()
        
#         if choice.lower() in ['quit', 'exit', 'q']:
#             sys.exit(0)
#         elif choice.lower() == 'status':
#             self.show_status()
#             return
#         elif choice.lower() == 'all':
#             selected = results
#         else:
#             try:
#                 selected_indices = []
#                 for part in choice.split(','):
#                     part = part.strip()
#                     if '-' in part:
#                         start, end = map(int, part.split('-'))
#                         selected_indices.extend(range(start-1, end))
#                     else:
#                         selected_indices.append(int(part) - 1)
                
#                 selected = [results[i] for i in selected_indices if 0 <= i < len(results)]
#             except:
#                 print("❌ Invalid selection")
#                 return
        
#         # View files
#         for i, result in enumerate(selected, 1):
#             print(f"\n📖 FILE {i}/{len(selected)}: {result['file_name']}")
#             print("=" * 100)
            
#             try:
#                 with open(result['file_path'], 'r', encoding='utf-8') as f:
#                     print(f.read())
#             except Exception as e:
#                 print(f"❌ Error: {e}")
            
#             print("=" * 100)
            
#             if i < len(selected):
#                 user_input = input("\nPress Enter for next file or 'quit' to exit: ").strip()
#                 if user_input.lower() in ['quit', 'exit', 'q']:
#                     sys.exit(0)
    
#     def show_status(self):
#         """Show system status"""
#         status = self.dynamic_vector_store.get_status()
        
#         print(f"\n📊 SYSTEM STATUS:")
#         print("=" * 50)
#         print(f"📚 Total chunks indexed: {status['total_chunks']}")
#         print(f"📄 Total files indexed: {status['total_files']}")
#         print(f"🔄 Currently updating: {'Yes' if status['is_updating'] else 'No'}")
#         print(f"⏳ Pending updates: {status['pending_updates']}")
#         print(f"👀 File monitoring: {'Active' if status['monitoring_active'] else 'Inactive'}")
#         print("=" * 50)
    
#     def interactive_search(self):
#         """Main interface with dynamic updates"""
#         print("\n🎉 Enhanced system ready!")
        
#         # Wait for any initial building to complete
#         while self.dynamic_vector_store.is_updating:
#             print("⏳ Building initial index, please wait...")
#             time.sleep(2)
        
#         while True:
#             status = self.dynamic_vector_store.get_status()
            
#             print("\n" + "=" * 80)
#             print("🚀 ENHANCED SMART DOCUMENT FINDER")
#             print("⚡ Keyword search • 🧠 Semantic search • 🔄 Dynamic updates")
            
#             # Show helpful message if no chunks indexed yet
#             if status['total_chunks'] == 0:
#                 md_files = list(self.vault_path.rglob("*.md"))
#                 if md_files:
#                     print(f"⚠️  No chunks indexed yet ({len(md_files)} files found)")
#                     print("💡 Try 'rebuild' to build the semantic index")
#                 else:
#                     print("📭 No markdown files found in vault")
#             else:
#                 print(f"📊 {status['total_chunks']} chunks from {status['total_files']} files")
            
#             if status['monitoring_active']:
#                 print("👀 Real-time monitoring: ACTIVE")
#             if status['is_updating']:
#                 print(f"🔄 Updating in background ({status['pending_updates']} pending)")
#             print("=" * 80)

#             query = input("\n💭 Enter your search query (or 'status', 'rebuild', 'quit'): ").strip()

#             if not query:
#                 continue

#             if query.lower() in ['quit', 'exit', 'q']:
#                 print("👋 Goodbye!")
#                 self.dynamic_vector_store.shutdown()
#                 break

#             elif query.lower() == 'status':
#                 self.show_status()
#                 continue

#             elif query.lower() == 'rebuild':
#                 print("🔄 Starting full rebuild...")
#                 self.dynamic_vector_store.force_full_rebuild()
#                 continue

#             # Perform smart search
#             results = self.smart_search(query)

#             # Display and handle results
#             self.display_results(results)
#             if results:
#                 self.select_and_view(results)


# if __name__ == "__main__":
#     try:
#         finder = EnhancedUnifiedFinder()
#         finder.interactive_search()
#     except KeyboardInterrupt:
#         print("\n👋 Interrupted. Goodbye!")
#     except Exception as e:
#         print(f"❌ Error: {e}")
from pathlib import Path
import sys
from typing import List, Dict
import time
import re
import shutil
from datetime import datetime
import os
import subprocess  # For opening files in default viewer

# Import components
try:
    from keyword_finder import FastKeywordFinder
    from vault_config import VaultConfig
    from dynamic_vector_store import DynamicVectorStore
    from image_handler import SimpleImageHandler  # Add this import
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False
    print("❌ Missing components")

class EnhancedUnifiedFinder:
    def __init__(self):
        if not COMPONENTS_AVAILABLE:
            print("❌ Cannot initialize. Missing required components.")
            sys.exit(1)

        # Initialize components
        print("⚡ Initializing enhanced document finder...")
        self.keyword_finder = FastKeywordFinder()
        self.vault_path = Path(self.keyword_finder.config.get_vault_path())

        # Initialize dynamic vector store
        print("🔄 Initializing dynamic vector store...")
        self.dynamic_vector_store = DynamicVectorStore(str(self.vault_path))
        
        # Initialize image handler
        print("🖼️ Initializing image handler...")
        self.image_handler = SimpleImageHandler(self.vault_path)

        # Wait a moment for initial setup
        time.sleep(1)

    def smart_search(self, query: str) -> List[Dict]:
        """Smart search with dynamic updates"""
        query = query.strip()
        word_count = len(query.split())

        # Show update status
        status = self.dynamic_vector_store.get_status()
        if status['is_updating']:
            print(f"🔄 (Updating in background - {status['pending_updates']} files pending)")

        if word_count <= 3:
            print(f"⚡ Using KEYWORD search ({word_count} words)")
            return self.keyword_finder.search(query)
        else:
            print(f"🧠 Using SEMANTIC search ({word_count} words)")
            results = self.dynamic_vector_store.search(query)
            if not results:
                print("🔄 No semantic results, trying keyword search...")
                key_terms = [w for w in query.split() if len(w) > 3][:2]
                if key_terms:
                    return self.keyword_finder.search(" ".join(key_terms))
            return results

    def display_results(self, results: List[Dict]):
        """Display search results with image info"""
        if not results:
            print("❌ No results found")
            return

        print(f"\n📋 FOUND {len(results)} DOCUMENTS:")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['file_name']}")
            print(f"   📂 {result['folder']}")
            print(f"   🔍 {result['search_type'].upper()}")
            
            if result.get('relevance_score'):
                print(f"   📈 Relevance: {result['relevance_score']:.3f}")
            
            # Check for images in this file
            image_info = self.image_handler.get_image_info_for_file(result['file_path'])
            if image_info:
                found_images = len([img for img in image_info if img['exists']])
                total_images = len(image_info)
                if found_images > 0:
                    print(f"   🖼️ Images: {found_images}/{total_images} found")
                else:
                    print(f"   🖼️ Images: {total_images} referenced (not found)")
            
            print(f"   📝 Preview:")
            context_lines = result['context'].split('\n')[:2]
            for line in context_lines:
                print(f"     {line[:100]}{'...' if len(line) > 100 else ''}")

    def select_and_view(self, results: List[Dict]):
        """Handle file selection and viewing with enhanced export options"""
        if not results:
            return

        choice = input(f"\nSelect files (1-{len(results)}, 'all', 'export', 'export-with-images', 'quit'): ").strip()

        if choice.lower() in ['quit', 'exit', 'q']:
            sys.exit(0)
        elif choice.lower() == 'status':
            self.show_status()
            return
        elif choice.lower() == 'export':
            # Quick export current results
            query = input("Enter query name for export: ").strip()
            if query:
                output_file = self.export_search_results_to_md(query)
                if output_file:
                    view_choice = input("Open exported file? (y/n): ").strip().lower()
                    if view_choice == 'y':
                        self.open_file_in_default_app(output_file)
            return
        elif choice.lower() == 'export-with-images':
            # Export with images copied
            query = input("Enter query name for export: ").strip()
            if query:
                output_file = self.export_search_results_with_images(query)
                if output_file:
                    view_choice = input("Open exported file? (y/n): ").strip().lower()
                    if view_choice == 'y':
                        self.open_file_in_default_app(output_file)
            return
        elif choice.lower() == 'all':
            selected = results
        else:
            try:
                selected_indices = []
                for part in choice.split(','):
                    part = part.strip()
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        selected_indices.extend(range(start-1, end))
                    else:
                        selected_indices.append(int(part) - 1)
                selected = [results[i] for i in selected_indices if 0 <= i < len(results)]
            except:
                print("❌ Invalid selection")
                return

        # View individual files
        for i, result in enumerate(selected, 1):
            print(f"\n📖 FILE {i}/{len(selected)}: {result['file_name']}")
            print("=" * 100)
            
            try:
                with open(result['file_path'], 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Show the processed content with image references
                processed_content = self.image_handler.process_content_with_images(content)
                print(processed_content)
                
                # Show image information
                image_info = self.image_handler.get_image_info_for_file(result['file_path'])
                if image_info:
                    print(f"\n🖼️ IMAGES REFERENCED IN THIS FILE:")
                    for img in image_info:
                        status = "✅ Found" if img['exists'] else "❌ Missing"
                        print(f"   {img['filename']} - {status}")
                        if img['exists']:
                            print(f"     Path: {img['path']}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                
            print("=" * 100)

            if i < len(selected):
                user_input = input("\nPress Enter for next, 'export' to save as markdown, 'export-img' for markdown+images, or 'quit': ").strip().lower()
                
                if user_input in ['quit', 'exit', 'q']:
                    sys.exit(0)
                elif user_input == 'export':
                    # Export this single file
                    filename = f"single_{result['file_name'].replace('.md', '')}"
                    temp_results = [result]
                    self._export_results_as_md(temp_results, filename)
                elif user_input == 'export-img':
                    # Export this single file with images
                    filename = f"single_{result['file_name'].replace('.md', '')}"
                    temp_results = [result]
                    self._export_results_with_images(temp_results, filename)

    def _export_results_as_md(self, results: List[Dict], query: str) -> str:
        """Helper method to export specific results"""
        safe_query = re.sub(r'[^\w\s-]', '', query.strip())[:50]
        safe_query = re.sub(r'[-\s]+', '_', safe_query)
        output_filename = f"rag_context_{safe_query}.md"
        output_path = self.vault_path / output_filename
        
        # Create markdown content
        markdown_content = f"# RAG Context: {query}\n\n"
        markdown_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for i, result in enumerate(results, 1):
            markdown_content += f"## Document {i}: {result['file_name']}\n\n"
            
            try:
                with open(result['file_path'], 'r', encoding='utf-8') as f:
                    content = f.read()
                processed_content = self.image_handler.process_content_with_images(content)
                markdown_content += processed_content + "\n\n---\n\n"
            except Exception as e:
                markdown_content += f"*Error reading file: {e}*\n\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"✅ Exported to: {output_path}")
        
        # Ask to open
        open_choice = input("Open the exported file? (y/n): ").strip().lower()
        if open_choice == 'y':
            self.open_file_in_default_app(str(output_path))
        
        return str(output_path)

    def _export_results_with_images(self, results: List[Dict], query: str) -> str:
        """Helper method to export specific results with images"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = re.sub(r'[^\w\s-]', '', query.strip())[:30].replace(' ', '_')
        output_folder = f"rag_context_{safe_query}_{timestamp}"
        
        output_path = self.vault_path / output_folder
        output_path.mkdir(exist_ok=True)
        images_folder = output_path / "images"
        images_folder.mkdir(exist_ok=True)
        
        # Create markdown content
        markdown_content = f"# RAG Context: {query}\n\n"
        
        for i, result in enumerate(results, 1):
            markdown_content += f"## Document {i}: {result['file_name']}\n\n"
            
            try:
                with open(result['file_path'], 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Copy images and update references
                image_info = self.image_handler.get_image_info_for_file(result['file_path'])
                processed_content = content
                
                for img in image_info:
                    if img['exists']:
                        source_path = Path(img['path'])
                        dest_path = images_folder / source_path.name
                        try:
                            shutil.copy2(source_path, dest_path)
                            old_ref = f"![[{img['filename']}]]"
                            new_ref = f"![{img['filename']}](images/{source_path.name})"
                            processed_content = processed_content.replace(old_ref, new_ref)
                        except Exception as e:
                            print(f"⚠️ Error copying {img['filename']}: {e}")
                
                markdown_content += processed_content + "\n\n---\n\n"
                
            except Exception as e:
                markdown_content += f"*Error reading file: {e}*\n\n"
        
        # Write markdown file
        md_file_path = output_path / "context.md"
        with open(md_file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"✅ Exported to: {md_file_path}")
        
        # Ask to open
        open_choice = input("Open the exported file? (y/n): ").strip().lower()
        if open_choice == 'y':
            self.open_file_in_default_app(str(md_file_path))
        
        return str(md_file_path)


    def show_status(self):
        """Show system status"""
        status = self.dynamic_vector_store.get_status()
        print(f"\n📊 SYSTEM STATUS:")
        print("=" * 50)
        print(f"📚 Total chunks indexed: {status['total_chunks']}")
        print(f"📄 Total files indexed: {status['total_files']}")
        print(f"🔄 Currently updating: {'Yes' if status['is_updating'] else 'No'}")
        print(f"⏳ Pending updates: {status['pending_updates']}")
        print(f"👀 File monitoring: {'Active' if status['monitoring_active'] else 'Inactive'}")
        print(f"🖼️ Images indexed: {len(self.image_handler.image_index)}")
        print("=" * 50)

    def interactive_search(self):
        """Main interface with dynamic updates"""
        print("\n🎉 Enhanced system ready!")

        # Wait for any initial building to complete
        while self.dynamic_vector_store.is_updating:
            print("⏳ Building initial index, please wait...")
            time.sleep(2)

        while True:
            status = self.dynamic_vector_store.get_status()
            print("\n" + "=" * 80)
            print("🚀 ENHANCED SMART DOCUMENT FINDER")
            print("⚡ Keyword search • 🧠 Semantic search • 🔄 Dynamic updates • 🖼️ Image support")

            # Show helpful message if no chunks indexed yet
            if status['total_chunks'] == 0:
                md_files = list(self.vault_path.rglob("*.md"))
                if md_files:
                    print(f"⚠️ No chunks indexed yet ({len(md_files)} files found)")
                    print("💡 Try 'rebuild' to build the semantic index")
                else:
                    print("📭 No markdown files found in vault")
            else:
                print(f"📊 {status['total_chunks']} chunks from {status['total_files']} files")
                print(f"🖼️ {len(self.image_handler.image_index)} images available")

            if status['monitoring_active']:
                print("👀 Real-time monitoring: ACTIVE")

            if status['is_updating']:
                print(f"🔄 Updating in background ({status['pending_updates']} pending)")

            print("=" * 80)

            query = input("\n💭 Enter your search query (or 'status', 'rebuild', 'quit'): ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                self.dynamic_vector_store.shutdown()
                break
            elif query.lower() == 'status':
                self.show_status()
                continue
            elif query.lower() == 'rebuild':
                print("🔄 Starting full rebuild...")
                self.dynamic_vector_store.force_full_rebuild()
                continue
            # In the interactive_search method, after the existing command checks, add:
            elif query.lower().startswith('debug '):
                filename = query[6:].strip()  # Remove 'debug ' prefix
                self.debug_image_search(filename)
                continue
            # Add this after existing command checks:
            elif query.lower().startswith('export '):
                search_query = query[7:].strip()  # Remove 'export ' prefix
                if search_query:
                    output_file = self.export_search_results_to_md(search_query)
                    if output_file:
                        print(f"📁 Context file created: {output_file}")
                        view_file = input("Open exported file? (y/n): ").strip().lower()
                        if view_file == 'y':
                            self.open_file_in_default_app(output_file)
                else:
                    print("❌ Please provide a search query after 'export'")
                continue

            elif query.lower().startswith('export-img '):
                search_query = query[11:].strip()  # Remove 'export-img ' prefix
                if search_query:
                    output_file = self.export_search_results_with_images(search_query)
                    if output_file:
                        view_file = input("Open exported file? (y/n): ").strip().lower()
                        if view_file == 'y':
                            self.open_file_in_default_app(output_file)
                else:
                    print("❌ Please provide a search query after 'export-img'")
                continue

            # Perform smart search
            results = self.smart_search(query)

            # Display and handle results
            self.display_results(results)
            if results:
                self.select_and_view(results)
    def debug_image_search(self, filename: str):
        """Debug image search for troubleshooting"""
        debug_info = self.image_handler.debug_search(filename)
        print(f"\n🔍 DEBUG: Searching for '{filename}'")
        print(f"   Exact match found: {debug_info['exact_match']}")
        print(f"   Timestamp extracted: {debug_info['timestamp_found']}")
        if debug_info['timestamp_found']:
            print(f"   Timestamp in index: {debug_info.get('timestamp_in_index', False)}")
        print(f"   Partial matches found: {len(debug_info['partial_matches'])}")
        for match in debug_info['partial_matches']:
            path = self.image_handler.find_image_path(match)
            print(f"     - {match} -> {path}")
        
        # Show all available timestamps for reference
        print(f"\n📊 Available timestamps in index:")
        timestamps = [key for key in self.image_handler.partial_name_index.keys()]
        for i, ts in enumerate(sorted(timestamps)[:10]):  # Show first 10
            print(f"     {ts}")
        if len(timestamps) > 10:
            print(f"     ... and {len(timestamps) - 10} more")
    def export_search_results_to_md(self, query: str, output_filename: str = None) -> str:
        """Export search results to a markdown file for RAG"""
        if not output_filename:
            # Create filename based on query
            safe_query = re.sub(r'[^\w\s-]', '', query.strip())[:50]
            safe_query = re.sub(r'[-\s]+', '_', safe_query)
            output_filename = f"rag_context_{safe_query}.md"
        
        output_path = self.vault_path / output_filename
        
        # Get search results
        results = self.smart_search(query)
        
        if not results:
            print("❌ No results found to export")
            return ""
        
        # Create comprehensive markdown context
        markdown_content = f"# RAG Context: {query}\n\n"
        markdown_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        markdown_content += f"**Query:** {query}\n"
        markdown_content += f"**Documents Found:** {len(results)}\n\n"
        markdown_content += "---\n\n"
        
        for i, result in enumerate(results, 1):
            # Add document header
            markdown_content += f"## Document {i}: {result['file_name']}\n\n"
            markdown_content += f"**Path:** `{result['file_path']}`\n"
            markdown_content += f"**Folder:** {result['folder']}\n"
            markdown_content += f"**Search Type:** {result['search_type']}\n"
            
            if result.get('relevance_score'):
                markdown_content += f"**Relevance Score:** {result['relevance_score']:.3f}\n"
            
            # Check for images
            image_info = self.image_handler.get_image_info_for_file(result['file_path'])
            if image_info:
                found_images = [img for img in image_info if img['exists']]
                markdown_content += f"**Images:** {len(found_images)}/{len(image_info)} found\n"
            
            markdown_content += "\n### Content:\n\n"
            
            # Read and process the actual file content
            try:
                with open(result['file_path'], 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Convert Obsidian image links to standard markdown
                processed_content = self.image_handler.process_content_with_images(content)
                markdown_content += processed_content
                
            except Exception as e:
                markdown_content += f"*Error reading file: {e}*\n"
            
            markdown_content += "\n\n---\n\n"
        
        # Write to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            print(f"✅ Context exported to: {output_path}")
            print(f"📊 {len(results)} documents with images ready for RAG")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Error writing context file: {e}")
            return ""

    def export_search_results_with_images(self, query: str, output_folder: str = None) -> str:
            """Export search results to a markdown file WITH images copied locally"""
            if not output_folder:
                # Create timestamped folder
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_query = re.sub(r'[^\w\s-]', '', query.strip())[:30].replace(' ', '_')
                output_folder = f"rag_context_{safe_query}_{timestamp}"
            
            output_path = self.vault_path / output_folder
            output_path.mkdir(exist_ok=True)
            
            # Create images subfolder
            images_folder = output_path / "images"
            images_folder.mkdir(exist_ok=True)
            
            results = self.smart_search(query)
            if not results:
                print("❌ No results found to export")
                return ""
            
            # Create markdown content with metadata
            markdown_content = f"# RAG Context: {query}\n\n"
            markdown_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            markdown_content += f"**Query:** {query}\n"
            markdown_content += f"**Documents Found:** {len(results)}\n\n"
            markdown_content += "---\n\n"
            
            for i, result in enumerate(results, 1):
                markdown_content += f"## Document {i}: {result['file_name']}\n\n"
                markdown_content += f"**Path:** `{result['file_path']}`\n"
                markdown_content += f"**Search Type:** {result['search_type']}\n"
                
                if result.get('relevance_score'):
                    markdown_content += f"**Relevance Score:** {result['relevance_score']:.3f}\n"
                
                markdown_content += "\n### Content:\n\n"
                
                # Read original content
                try:
                    with open(result['file_path'], 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Find and copy images
                    image_info = self.image_handler.get_image_info_for_file(result['file_path'])
                    processed_content = content
                    
                    for img in image_info:
                        if img['exists']:
                            # Copy image to local images folder
                            source_path = Path(img['path'])
                            dest_path = images_folder / source_path.name
                            
                            try:
                                shutil.copy2(source_path, dest_path)
                                # Replace with local reference
                                old_ref = f"![[{img['filename']}]]"
                                new_ref = f"![{img['filename']}](images/{source_path.name})"
                                processed_content = processed_content.replace(old_ref, new_ref)
                                print(f"📋 Copied image: {img['filename']}")
                            except Exception as e:
                                print(f"⚠️ Error copying {img['filename']}: {e}")
                    
                    markdown_content += processed_content + "\n\n---\n\n"
                    
                except Exception as e:
                    markdown_content += f"*Error reading file: {e}*\n\n"
            
            # Write markdown file
            md_file_path = output_path / "context.md"
            with open(md_file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            print(f"✅ Context exported to: {output_path}")
            print(f"📁 Markdown file: {md_file_path}")
            print(f"🖼️ Images folder: {images_folder}")
            print(f"📖 Open {md_file_path} in Typora/VS Code to see images!")
            
            return str(md_file_path)

    def open_file_in_default_app(self, file_path: str) -> bool:
            """Open file in default application"""
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(file_path)
                elif os.name == 'posix':  # macOS and Linux
                    subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', file_path])
                return True
            except Exception as e:
                print(f"❌ Error opening file: {e}")
                return False

    def quick_export(self, query: str) -> str:
            """Quick export with automatic filename"""
            return self.export_search_results_to_md(query)

if __name__ == "__main__":
    try:
        finder = EnhancedUnifiedFinder()
        finder.interactive_search()
    except KeyboardInterrupt:
        print("\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
