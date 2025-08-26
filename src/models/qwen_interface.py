import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

class Qwen3Interface:
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Thinking-2507"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.max_context = 131072  # Safe context length
        self.load_model()
    
    def load_model(self):
        """Load the Qwen3-4B model"""
        print(f"Loading {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Configure for thinking mode
        if hasattr(self.model.config, 'max_position_embeddings'):
            actual_max = self.model.config.max_position_embeddings
            self.max_context = min(self.max_context, actual_max)
        
        print(f"✅ Model loaded. Context length: {self.max_context}")
    
    def generate(self, prompt: str, max_tokens: int = 1024, 
                temperature: float = 0.6) -> str:
        """Generate response from the model"""
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True,
            max_length=self.max_context - max_tokens
        )
        
        # Generate response  
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                top_k=20,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        return len(self.tokenizer.encode(text))