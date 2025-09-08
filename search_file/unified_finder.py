# from pathlib import Path
# import sys
# from typing import List, Dict
# import time
# import re
# import shutil
# from datetime import datetime
# import os
# import subprocess

# # Import components
# try:
#     from keyword_finder import FastKeywordFinder
#     from vault_config import VaultConfig
#     from dynamic_vector_store import DynamicVectorStore
#     from image_handler import SimpleImageHandler
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
        
#         # Initialize image handler
#         print("🖼️ Initializing image handler...")
#         self.image_handler = SimpleImageHandler(self.vault_path)

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
#         """Display search results with image info"""
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
            
#             # Check for images in this file
#             image_info = self.image_handler.get_image_info_for_file(result['file_path'])
#             if image_info:
#                 found_images = len([img for img in image_info if img['exists']])
#                 total_images = len(image_info)
#                 if found_images > 0:
#                     print(f"   🖼️ Images: {found_images}/{total_images} found")
#                 else:
#                     print(f"   🖼️ Images: {total_images} referenced (not found)")
            
#             print(f"   📝 Preview:")
#             context_lines = result['context'].split('\n')[:2]
#             for line in context_lines:
#                 print(f"     {line[:100]}{'...' if len(line) > 100 else ''}")

#     def select_and_view(self, results: List[Dict]):
#         """Handle file selection and viewing - export options come after viewing each document"""
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

#         # View individual files with export options
#         for i, result in enumerate(selected, 1):
#             print(f"\n📖 FILE {i}/{len(selected)}: {result['file_name']}")
#             print("=" * 100)
            
#             try:
#                 with open(result['file_path'], 'r', encoding='utf-8') as f:
#                     content = f.read()
                
#                 # Show the processed content with image references
#                 processed_content = self.image_handler.process_content_with_images(content)
#                 print(processed_content)
                
#                 # Show image information
#                 image_info = self.image_handler.get_image_info_for_file(result['file_path'])
#                 if image_info:
#                     print(f"\n🖼️ IMAGES REFERENCED IN THIS FILE:")
#                     for img in image_info:
#                         status = "✅ Found" if img['exists'] else "❌ Missing"
#                         print(f"   {img['filename']} - {status}")
#                         if img['exists']:
#                             print(f"     Path: {img['path']}")
                
#             except Exception as e:
#                 print(f"❌ Error: {e}")
                
#             print("=" * 100)

#             # Export options for each individual document
#             if i < len(selected):
#                 user_input = input("\nPress Enter for next file, 'export' to save this document with images, or 'quit': ").strip().lower()
                
#                 if user_input in ['quit', 'exit', 'q']:
#                     sys.exit(0)
#                 elif user_input == 'export':
#                     # Export ONLY this single document with its images
#                     self.export_single_document_with_images(result)
#             else:
#                 # Last file - ask for export
#                 user_input = input("\n'export' to save this document with images, or Enter to continue: ").strip().lower()
                
#                 if user_input == 'export':
#                     # Export ONLY this single document with its images
#                     self.export_single_document_with_images(result)

#     def export_single_document_with_images(self, result: Dict):
#         """Export the selected single document with ONLY its images"""
#         # Create unique RAG folder name in the specified export path
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         doc_name = result['file_name'].replace('.md', '').replace(' ', '_')
#         safe_name = re.sub(r'[^\w\s-]', '', doc_name)[:30]
#         rag_folder = self.rag_export_path / f"rag_{safe_name}_{timestamp}"
#         images_folder = rag_folder / "images"
#         rag_folder.mkdir(exist_ok=True)
#         images_folder.mkdir(exist_ok=True)

#         attachment_root = Path(r"C:/Users/Lenovo/Documents/Obsidian Vault/1 - Attachments")

#         # Process document
#         with open(result['file_path'], 'r', encoding='utf-8') as f:
#             content = f.read()

#         # Find image references
#         img_pattern = r'!\[\[([^|\]]+\.(?:png|jpg|jpeg|gif|svg|webp|bmp))\]\]'
#         image_refs = re.findall(img_pattern, content, re.IGNORECASE)

#         images_copied = 0
#         for img_filename in set(image_refs):
#             actual_path = None
#             # Search for exact match or by timestamp under attachments
#             for root, dirs, files in os.walk(attachment_root):
#                 for file in files:
#                     # Exact filename match
#                     if file == img_filename:
#                         actual_path = Path(root) / file
#                         break
#                     # Timestamp based match
#                     num_pat = re.search(r'(\d{14})', img_filename)
#                     if num_pat and file.endswith(Path(img_filename).suffix):
#                         timestamp = num_pat.group(1)
#                         if timestamp in file:
#                             actual_path = Path(root) / file
#                             break
#                 if actual_path:
#                     break
#             if actual_path:
#                 shutil.copy2(actual_path, images_folder / actual_path.name)
#                 images_copied += 1
#             else:
#                 print(f"❌ Image '{img_filename}' NOT found in attachments.")

#         # Replace Obsidian refs with local image links
#         def img_repl(match):
#             filename = match.group(1)
#             return f"![{filename}](images/{filename})"
#         processed_content = re.sub(img_pattern, img_repl, content)

#         # Write the context.md
#         md_file_path = rag_folder / "context.md"
#         with open(md_file_path, 'w', encoding='utf-8') as f:
#             f.write(processed_content)

#         print(f"\n✅ Document exported to: {rag_folder}")
#         print(f"📁 Markdown file: {md_file_path}")
#         print(f"🖼️ Images copied: {images_copied}")
#         print(f"📖 Open {md_file_path} in Typora/VS Code to see images!")
#         print("✨ Export complete! You can continue searching...\n")


#     def open_file_in_default_app(self, file_path: str) -> bool:
#         """Open file in default application"""
#         try:
#             if os.name == 'nt':  # Windows
#                 os.startfile(file_path)
#             elif os.name == 'posix':  # macOS and Linux
#                 subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', file_path])
#             return True
#         except Exception as e:
#             print(f"❌ Error opening file: {e}")
#             return False

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
#         print(f"🖼️ Images indexed: {len(self.image_handler.image_index)}")
#         print("=" * 50)

#     def interactive_search(self):
#         """Main interface - search first, then view, then export individual documents"""
#         print("\n🎉 Enhanced system with images ready!")

#         # Wait for any initial building to complete
#         while self.dynamic_vector_store.is_updating:
#             print("⏳ Building initial index, please wait...")
#             time.sleep(2)

#         while True:
#             status = self.dynamic_vector_store.get_status()
#             print("\n" + "=" * 80)
#             print("🚀 ENHANCED SMART DOCUMENT FINDER")
#             print("⚡ Keyword search • 🧠 Semantic search • 🔄 Dynamic updates • 🖼️ Image support")

#             # Show helpful message if no chunks indexed yet
#             if status['total_chunks'] == 0:
#                 md_files = list(self.vault_path.rglob("*.md"))
#                 if md_files:
#                     print(f"⚠️ No chunks indexed yet ({len(md_files)} files found)")
#                     print("💡 Try 'rebuild' to build the semantic index")
#                 else:
#                     print("📭 No markdown files found in vault")
#             else:
#                 print(f"📊 {status['total_chunks']} chunks from {status['total_files']} files")
#                 print(f"🖼️ {len(self.image_handler.image_index)} images available")

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

#             # Display and handle results - export happens during viewing individual documents
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
import subprocess

# Import components
try:
    from keyword_finder import FastKeywordFinder
    from vault_config import VaultConfig
    from dynamic_vector_store import DynamicVectorStore
    from image_handler import SimpleImageHandler
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

        # FIX: Add the missing rag_export_path attribute
        self.rag_export_path = Path(r"E:\7. Projects From Sem 3\RAG\data")
        self.rag_export_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 RAG export path set to: {self.rag_export_path}")

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
        """Handle file selection and viewing - export options come after viewing each document"""
        if not results:
            return

        choice = input(f"\nSelect files (1-{len(results)}, ranges, 'all', 'status', 'quit'): ").strip()

        if choice.lower() in ['quit', 'exit', 'q']:
            sys.exit(0)
        elif choice.lower() == 'status':
            self.show_status()
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

        # View individual files with export options
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

            # Export options for each individual document
            if i < len(selected):
                user_input = input("\nPress Enter for next file, 'export' to save this document with images, or 'quit': ").strip().lower()
                
                if user_input in ['quit', 'exit', 'q']:
                    return  # Return to main menu
                elif user_input == 'export':
                    # Export ONLY this single document with its images
                    self.export_single_document_with_images(result)
            else:
                # Last file - ask for export
                user_input = input("\n'export' to save this document with images, or Enter to continue: ").strip().lower()
                
                if user_input == 'export':
                    # Export ONLY this single document with its images
                    self.export_single_document_with_images(result)

    def export_single_document_with_images(self, result: Dict):
        """Export ONLY the selected single document with its images to RAG folder"""
        
        # Create unique RAG folder name in the specified export path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        document_name = result['file_name'].replace('.md', '').replace(' ', '_')
        safe_name = re.sub(r'[^\w\s-]', '', document_name)[:30]
        rag_folder_name = f"rag_{safe_name}_{timestamp}"
        
        # Use the specified export path
        rag_folder = self.rag_export_path / rag_folder_name
        rag_folder.mkdir(exist_ok=True)
        images_folder = rag_folder / "images"
        images_folder.mkdir(exist_ok=True)
        
        print(f"📁 Creating RAG folder: {rag_folder}")
        
        # Fixed attachments path
        attachment_root = Path(r"C:\Users\Lenovo\Documents\Obsidian Vault\1 - Attachments")
        
        # Read and process the SINGLE document content
        try:
            with open(result['file_path'], 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find all image references in the document
            img_pattern = r'!\[\[([^|\]]+\.(?:png|jpg|jpeg|gif|svg|webp|bmp))\]\]'
            image_refs = re.findall(img_pattern, content, re.IGNORECASE)
            
            print(f"🔍 Found {len(image_refs)} image references in document")
            
            images_copied = 0
            for img_filename in set(image_refs):  # Remove duplicates
                print(f"🔎 Looking for: {img_filename}")
                
                actual_path = None
                # Search for the image in attachments folder
                for root, dirs, files in os.walk(attachment_root):
                    for file in files:
                        # Try exact filename match
                        if file == img_filename:
                            actual_path = Path(root) / file
                            break
                        
                        # Try timestamp-based match for pasted images
                        timestamp_match = re.search(r'(\d{14})', img_filename)
                        if timestamp_match and file.endswith(Path(img_filename).suffix):
                            timestamp = timestamp_match.group(1)
                            if timestamp in file:
                                actual_path = Path(root) / file
                                break
                    
                    if actual_path:
                        break
                
                # Copy the image if found
                if actual_path and actual_path.exists():
                    try:
                        dest_path = images_folder / actual_path.name
                        shutil.copy2(actual_path, dest_path)
                        images_copied += 1
                        print(f"✅ Copied: {img_filename}")
                    except Exception as e:
                        print(f"❌ Failed to copy {img_filename}: {e}")
                else:
                    print(f"❌ Not found: {img_filename}")
            
            # Replace Obsidian image links with standard markdown
            def replace_image_ref(match):
                filename = match.group(1)
                return f"![{filename}](images/{filename})"
            
            processed_content = re.sub(img_pattern, replace_image_ref, content, flags=re.IGNORECASE)
            
            # Create markdown content for SINGLE document
            markdown_content = f"# RAG Document: {result['file_name']}\n\n"
            markdown_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            markdown_content += f"**Source:** `{result['file_path']}`\n"
            markdown_content += f"**Folder:** {result['folder']}\n"
            
            if result.get('relevance_score'):
                markdown_content += f"**Relevance Score:** {result['relevance_score']:.3f}\n"
            
            markdown_content += f"**Images:** {images_copied} images copied\n\n"
            markdown_content += "---\n\n"
            
            # Add the processed content
            markdown_content += processed_content
            
        except Exception as e:
            markdown_content = f"# Error Reading Document\n\n*Error: {e}*\n"
            images_copied = 0
        
        # Write markdown file
        md_file_path = rag_folder / "context.md"
        with open(md_file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"✅ Document exported to: {rag_folder}")
        print(f"📁 Markdown file: {md_file_path}")
        print(f"🖼️ Images copied: {images_copied}")
        
        # Open the RAG data folder in existing VSCode window
        try:
            subprocess.Popen(["code", "--reuse-window", str(self.rag_export_path)], shell=True)
            print(f"🖥️ Opened RAG data folder in VSCode: {self.rag_export_path}")
        except Exception as e:
            print(f"⚠️ Could not open VSCode: {e}")
            print(f"📖 You can manually open: {rag_folder}")
        
        print("✨ Export complete! You can continue searching...\n")

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
        print(f"📁 RAG export path: {self.rag_export_path}")
        print("=" * 50)

    def interactive_search(self):
        """Main interface - search first, then view, then export individual documents"""
        print("\n🎉 Enhanced system with images ready!")

        # Wait for any initial building to complete
        while self.dynamic_vector_store.is_updating:
            print("⏳ Building initial index, please wait...")
            time.sleep(2)

        while True:
            try:
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

                # Perform smart search
                results = self.smart_search(query)

                # Display and handle results - export happens during viewing individual documents
                self.display_results(results)
                if results:
                    self.select_and_view(results)
                    
            except KeyboardInterrupt:
                print("\n\n🛑 Interrupted by user")
                confirm = input("Really quit? (y/n): ").strip().lower()
                if confirm == 'y':
                    print("👋 Goodbye!")
                    self.dynamic_vector_store.shutdown()
                    break
                else:
                    print("Continuing...")
                    continue
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                print("🔄 Returning to search prompt...")
                continue

if __name__ == "__main__":
    try:
        finder = EnhancedUnifiedFinder()
        finder.interactive_search()
    except KeyboardInterrupt:
        print("\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
