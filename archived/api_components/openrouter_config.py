"""
OpenRouter API Configuration for GEPA optimization with Groq provider
"""
import os
from typing import Dict, Any

class OpenRouterConfig:
    """Configuration for OpenRouter API models with Groq provider"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
    
    # Model configurations for GEPA+ART+RULER
    LOCAL_MODEL = "Qwen/Qwen3-4B-Thinking-2507"  # Local GPU model being trained
    REFLECTION_MODEL = "anthropic/claude-sonnet-4"  # OpenRouter: GEPA reflection
    JUDGE_MODEL = "anthropic/claude-sonnet-4"  # OpenRouter: LLM-as-Judge for RULER scoring
    
    # API settings
    BASE_URL = "https://openrouter.ai/api/v1"
    
    # Budget controls - optimized for competitive programming
    MAX_TOKENS_PER_REQUEST = 6000  # Sufficient for complex problem solving with multi-attempt reasoning
    DEFAULT_TEMPERATURE = 0.6
    DEFAULT_TOP_P = 0.95
    
    # Reasoning token settings
    REASONING_ENABLED = True
    REASONING_EFFORT = "high"  # high, medium, or low
    REASONING_MAX_TOKENS = 12000  # For models that support direct token allocation
    
    # Rate limiting
    REQUESTS_PER_MINUTE = 50
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    
    # Groq provider configuration
    GROQ_PROVIDER_CONFIG = {
        "provider": {
            "only": ["groq"],  # Only use Groq provider
            "allow_fallbacks": False,  # Disable fallbacks to other providers
            "data_collection": "deny"  # Use providers that don't collect data
        }
    }
    
    @classmethod
    def get_model_config(cls, model_name: str, use_groq_only: bool = True) -> Dict[str, Any]:
        """Get configuration for specific model"""
        base_config = {
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "base_url": cls.BASE_URL,
            "max_tokens": cls.MAX_TOKENS_PER_REQUEST,
            "temperature": cls.DEFAULT_TEMPERATURE,
            "top_p": cls.DEFAULT_TOP_P
        }
        
        # Add Groq-only provider configuration for GPT-OSS-20B
        if use_groq_only and "gpt-oss-20b" in model_name:
            base_config.update(cls.GROQ_PROVIDER_CONFIG)
        
        # Add reasoning configuration for supported models
        if cls.REASONING_ENABLED and "gpt-oss" in model_name:
            base_config["reasoning"] = {
                "effort": cls.REASONING_EFFORT,
                "max_tokens": cls.REASONING_MAX_TOKENS
            }
        
        # Model-specific configurations
        if "qwen3-4b-thinking" in model_name:
            base_config.update({
                "temperature": 0.6,  # Balanced for competitive programming
                "max_tokens": 6000,  # Support for complex reasoning + code + multi-attempts
                "top_p": 0.95
            })
        elif "gpt-oss-20b" in model_name:
            base_config.update({
                "temperature": 0.6,  # Slightly creative for problem solving
                "max_tokens": 6000   # Allow longer responses for complex problems
            })
        elif "claude" in model_name:
            base_config.update({
                "temperature": 0.7,  # More creative for reflection/judging
                "max_tokens": 4000   # Shorter for judging tasks
            })
            
        return base_config
    
    @classmethod
    def get_groq_specific_config(cls) -> Dict[str, Any]:
        """Get Groq-specific configuration for maximum performance"""
        return {
            "provider": {
                "only": ["groq"],
                "allow_fallbacks": False,
                "sort": "throughput",  # Prioritize throughput (Groq's strength)
                "require_parameters": True,  # Only use providers supporting all params
                "data_collection": "deny"
            }
        }
    
    @classmethod
    def create_groq_request_body(cls, messages: list, model: str = "openai/gpt-oss-20b") -> Dict[str, Any]:
        """Create a complete request body for Groq provider"""
        request_body = {
            "model": model,
            "messages": messages,
            "max_tokens": cls.MAX_TOKENS_PER_REQUEST,
            "temperature": cls.DEFAULT_TEMPERATURE,
            "top_p": cls.DEFAULT_TOP_P,
            **cls.get_groq_specific_config()
        }
        
        # Add reasoning for GPT-OSS models
        if cls.REASONING_ENABLED and "gpt-oss" in model:
            request_body["reasoning"] = {
                "effort": cls.REASONING_EFFORT,
                "max_tokens": cls.REASONING_MAX_TOKENS
            }
        
        return request_body
    
    @classmethod  
    def get_art_judge_config(cls, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get configuration for Claude Sonnet as ART RULER judge"""
        return {
            "model": cls.JUDGE_MODEL,
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "base_url": cls.BASE_URL,
            "max_tokens": 8000,  # Shorter responses for judgment
            "temperature": 0.3,  # Lower temp for consistent scoring
            "top_p": 0.9,
            "messages": [
                {
                    "role": "system", 
                    "content": """You are an expert judge for competitive programming solutions using the RULER methodology. 

Evaluate this trajectory on a scale of 0.0 to 1.0 considering:
1. **Solution Correctness**: Did it solve the problem correctly?
2. **Reasoning Quality**: Was the thinking process logical and thorough?  
3. **Code Quality**: Is the implementation clean and efficient?
4. **Error Recovery**: How well did it handle and learn from failures?
5. **Tool Usage**: Was OJBench evaluation used effectively?

Provide:
- Overall score (0.0-1.0)
- Detailed breakdown by category
- Specific improvement suggestions
- Comparison to typical solutions

Be consistent and fair in your scoring."""
                },
                {
                    "role": "user",
                    "content": f"""Please evaluate this competitive programming trajectory:

**Problem**: {trajectory_data.get('problem_statement', 'N/A')}
**Difficulty**: {trajectory_data.get('problem_difficulty', 'unknown')}
**Language**: {trajectory_data.get('language', 'unknown')}

**Solution Code**:
```{trajectory_data.get('language', 'text')}
{trajectory_data.get('generated_code', 'No code provided')}
```

**Reasoning Blocks**:
{trajectory_data.get('think_blocks_text', 'No reasoning provided')}

**Result**: {trajectory_data.get('verdict', 'unknown')} - {'Success' if trajectory_data.get('success', False) else 'Failed'}
**Test Cases**: {trajectory_data.get('test_cases_passed', 0)}/{trajectory_data.get('total_test_cases', 0)} passed
**Attempts**: {trajectory_data.get('attempt_number', 1)}/{trajectory_data.get('max_attempts', 3)}
**Solve Time**: {trajectory_data.get('total_solve_time', 0):.1f}s

Provide your evaluation as JSON with this structure:
```json
{{
  "overall_score": 0.0,
  "category_scores": {{
    "correctness": 0.0,
    "reasoning_quality": 0.0, 
    "code_quality": 0.0,
    "error_recovery": 0.0,
    "tool_usage": 0.0
  }},
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"],
  "improvement_suggestions": ["suggestion1", "suggestion2"],
  "reasoning": "Brief explanation of the scoring rationale"
}}
```"""
                }
            ]
        }

def setup_openrouter_env():
    """Setup environment variables for OpenRouter"""
    # First try to load from .env file
    from pathlib import Path
    
    env_file = Path(__file__).parent.parent.parent / ".env"
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value
                        
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("⚠️  OPENROUTER_API_KEY not found!")
        print("Please set it in your environment:")
        print("export OPENROUTER_API_KEY='your_key_here'")
        print("Or create a .env file with: OPENROUTER_API_KEY=your_key_here")
        return False
    
    print(f"✅ OpenRouter API configured with Groq provider preference")
    return True

# Example usage function
def example_groq_request():
    """Example of how to make a request using only Groq provider"""
    import requests
    
    config = OpenRouterConfig()
    
    # Create request body that only uses Groq
    request_body = config.create_groq_request_body(
        messages=[
            {"role": "system", "content": "You are a helpful assistant optimized for reasoning tasks."},
            {"role": "user", "content": "Explain the concept of mixture of experts in neural networks."}
        ],
        model="openai/gpt-oss-20b"
    )
    
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json"
    }
    
    print("Request configuration:")
    print(f"Model: {request_body['model']}")
    print(f"Provider restrictions: {request_body['provider']}")
    print(f"Reasoning enabled: {'reasoning' in request_body}")
    
    # Uncomment to make actual request:
    # response = requests.post(f"{config.BASE_URL}/chat/completions", 
    #                         json=request_body, headers=headers)
    # return response.json()

if __name__ == "__main__":
    setup_openrouter_env()
    example_groq_request()