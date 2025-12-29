"""
File discovery utilities
"""

from pathlib import Path
from config import IMAGE_EXTENSIONS, MARKDOWN_EXTENSIONS

def discover_files(path_str: str) -> tuple[list[str], list[str]]:
    """Find markdown files and images"""
    path_str = path_str.strip().strip('"').strip("'")
    path = Path(path_str)
    
    md_files = []
    image_files = []
    
    if not path.exists():
        return [], []
    
    if path.is_file():
        if path.suffix.lower() in MARKDOWN_EXTENSIONS:
            md_files.append(str(path.absolute()))
        elif path.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(str(path.absolute()))
        return md_files, image_files
    
    if path.is_dir():
        for pattern in ['*.md', '*.markdown']:
            md_files.extend(path.rglob(pattern))
        
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(path.rglob(f'*{ext}'))
            image_files.extend(path.rglob(f'*{ext.upper()}'))
        
        return [str(f.absolute()) for f in md_files], [str(f.absolute()) for f in image_files]
    
    return [], []


def validate_input_files(file_paths: list[str]) -> tuple[list[str], list[str]]:
    """Validate files"""
    all_md_files = []
    all_image_files = []
    
    for path_str in file_paths:
        md_files, img_files = discover_files(path_str)
        
        if md_files or img_files:
            all_md_files.extend(md_files)
            all_image_files.extend(img_files)
    
    return all_md_files, all_image_files
