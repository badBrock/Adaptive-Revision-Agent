"""
Keyword and image extractors
"""

import base64
from pathlib import Path
from rake_nltk import Rake
from groq import Groq

class KeywordExtractor:
    """Extract keywords using RAKE"""
    
    def __init__(self):
        self.rake = Rake()
    
    def extract(self, text: str, top_n: int = 10) -> list[str]:
        self.rake.extract_keywords_from_text(text)
        ranked_phrases = self.rake.get_ranked_phrases()
        return ranked_phrases[:top_n]


class ImageDescriptor:
    """Describe images using vision model"""
    
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int, prompt: str):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt = prompt
    
    def describe_image(self, image_path: str) -> str:
        try:
            if not Path(image_path).exists():
                return ""
            
            with open(image_path, "rb") as f:
                image_data = base64.standard_b64encode(f.read()).decode("utf-8")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                }],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
        except:
            return ""
