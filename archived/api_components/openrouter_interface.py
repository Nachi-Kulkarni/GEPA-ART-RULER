"""
OpenRouter API interface for GEPA optimization
Replaces local GPU model with API-based models
"""
import time
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.openrouter_config import OpenRouterConfig

@dataclass
class APIResponse:
    content: str
    usage: Dict[str, int]
    model: str
    success: bool
    error: Optional[str] = None

class OpenRouterInterface:
    """Interface for OpenRouter API models"""
    
    def __init__(self, model_name: str = OpenRouterConfig.TASK_MODEL):
        self.model_name = model_name
        self.config = OpenRouterConfig.get_model_config(model_name)
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/gepa-ai/gepa",
            "X-Title": "GEPA Optimization"
        })
        
    def generate(self, prompt: str, max_tokens: int = None, 
                temperature: float = None, **kwargs) -> APIResponse:
        """Generate response from OpenRouter model"""
        
        # Use provided parameters or defaults
        max_tokens = max_tokens or self.config["max_tokens"]
        temperature = temperature or self.config["temperature"]
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": self.config["top_p"],
            **kwargs
        }
        
        for attempt in range(OpenRouterConfig.MAX_RETRIES):
            try:
                response = self.session.post(
                    f"{self.config['base_url']}/chat/completions",
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return APIResponse(
                        content=data["choices"][0]["message"]["content"],
                        usage=data.get("usage", {}),
                        model=data.get("model", self.model_name),
                        success=True
                    )
                elif response.status_code == 429:  # Rate limit
                    wait_time = OpenRouterConfig.RETRY_DELAY * (2 ** attempt)
                    print(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    error_msg = f"API error {response.status_code}: {response.text}"
                    return APIResponse(
                        content="",
                        usage={},
                        model=self.model_name,
                        success=False,
                        error=error_msg
                    )
                    
            except Exception as e:
                if attempt == OpenRouterConfig.MAX_RETRIES - 1:
                    return APIResponse(
                        content="",
                        usage={},
                        model=self.model_name,
                        success=False,
                        error=str(e)
                    )
                time.sleep(OpenRouterConfig.RETRY_DELAY)
        
        return APIResponse(
            content="",
            usage={},
            model=self.model_name,
            success=False,
            error="Max retries exceeded"
        )
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4
    
    def test_connection(self) -> bool:
        """Test if API connection works"""
        try:
            response = self.generate("Test message", max_tokens=10)
            return response.success
        except Exception:
            return False

class OpenRouterGEPAAdapter:
    """Adapter to make OpenRouter interface compatible with GEPA"""
    
    def __init__(self, task_model: str = None, reflection_model: str = None):
        self.task_model = OpenRouterInterface(
            task_model or OpenRouterConfig.TASK_MODEL
        )
        self.reflection_model = OpenRouterInterface(
            reflection_model or OpenRouterConfig.REFLECTION_MODEL
        )
        
    def generate_solution(self, prompt: str) -> str:
        """Generate solution using task model"""
        response = self.task_model.generate(prompt, max_tokens=2048)
        if response.success:
            return response.content
        else:
            raise Exception(f"Task model error: {response.error}")
    
    def generate_reflection(self, prompt: str) -> str:
        """Generate reflection using reflection model"""
        response = self.reflection_model.generate(prompt, max_tokens=2048)
        if response.success:
            return response.content
        else:
            raise Exception(f"Reflection model error: {response.error}")
    
    def test_models(self) -> Dict[str, bool]:
        """Test both models"""
        return {
            "task_model": self.task_model.test_connection(),
            "reflection_model": self.reflection_model.test_connection()
        }

# Utility function for litellm compatibility
def create_litellm_function(model_name: str):
    """Create a litellm-compatible function for GEPA"""
    interface = OpenRouterInterface(model_name)
    
    def llm_function(prompt: str) -> str:
        response = interface.generate(prompt)
        if response.success:
            return response.content
        else:
            raise Exception(f"OpenRouter API error: {response.error}")
    
    return llm_function

if __name__ == "__main__":
    # Test the interface
    from ..config.openrouter_config import setup_openrouter_env
    
    if setup_openrouter_env():
        print("\n🧪 Testing OpenRouter interface...")
        
        adapter = OpenRouterGEPAAdapter()
        results = adapter.test_models()
        
        print(f"Task model ({OpenRouterConfig.TASK_MODEL}): {'✅' if results['task_model'] else '❌'}")
        print(f"Reflection model ({OpenRouterConfig.REFLECTION_MODEL}): {'✅' if results['reflection_model'] else '❌'}")
        
        if all(results.values()):
            print("\n🎉 All models working! Ready for GEPA optimization.")
        else:
            print("\n⚠️  Some models failed. Check your API key and model names.")