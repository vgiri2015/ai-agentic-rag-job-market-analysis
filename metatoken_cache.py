"""
MetaToken Semantic Cache Integration
Auto-generated integration code for ai-agentic-rag-job-market-analysis
"""

import os
import requests
from typing import Optional

class MetaTokenCache:
    """
    Semantic cache proxy for LLM calls
    Routes requests through MetaToken cache service
    """
    
    def __init__(self, user_token: str = None):
        self.user_token = user_token or os.environ.get("METATOKEN_USER_TOKEN")
        self.cache_endpoint = "https://ai-budget-guardian.preview.emergentagent.com/api/cache/query"
        
        if not self.user_token:
            raise ValueError("METATOKEN_USER_TOKEN environment variable required")
    
    def query(self, prompt: str, model: str = "gpt-4o-mini", 
              temperature: float = 0.7, max_tokens: int = None) -> dict:
        """
        Query LLM through MetaToken cache
        Returns: {"response": str, "cached": bool, "cost_saved": float}
        """
        try:
            payload = {
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "user_token": self.user_token
            }
            
            response = requests.post(self.cache_endpoint, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            # Log cache performance
            if result.get("cached"):
                print(f"✓ Cache HIT - Saved ${result.get('cost_saved', 0):.4f}")
            else:
                print(f"✗ Cache MISS - Response cached for future use")
            
            return result
            
        except Exception as e:
            print(f"MetaToken cache error: {e}")
            raise
    
    def chat_completion(self, messages: list, model: str = "gpt-4o-mini", **kwargs):
        """
        OpenAI-compatible chat completion interface
        """
        # Convert messages to single prompt
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        result = self.query(
            prompt=prompt,
            model=model,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens")
        )
        
        # Return OpenAI-compatible format
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": result["response"]
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "cached": result.get("cached", False),
                "cost_saved": result.get("cost_saved", 0)
            }
        }


# Usage Examples:

# Initialize cache
cache = MetaTokenCache(user_token="mtk_metatoken_8bC-A20TaTXi5EC_ZqZYBw")

# Example 1: Direct query
result = cache.query(
    prompt="Explain quantum computing in simple terms",
    model="gpt-4o-mini"
)
print(result["response"])

# Example 2: OpenAI-compatible interface
response = cache.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is machine learning?"}
    ],
    model="gpt-4o-mini"
)
print(response["choices"][0]["message"]["content"])
