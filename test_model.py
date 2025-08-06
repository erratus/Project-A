#!/usr/bin/env python3
"""
Simple test script to verify Ollama model is working
"""

import json
import time
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage

def test_model(model_name):
    """Test a single model with a simple prompt"""
    print(f"Testing {model_name}...")
    
    try:
        # Create model instance
        model = ChatOllama(
            model=model_name,
            temperature=0.0,
            timeout=60,  # 1 minute timeout
            num_ctx=1024,  # Smaller context
            num_predict=200,  # Limit output
        )
        
        # Simple test prompt
        prompt = """You are a resume-JD matcher. Return only this JSON:
{
  "test": {
    "match_pct": 85.5,
    "explanation": "This is a test response"
  }
}

Just return the JSON above, nothing else."""
        
        start_time = time.time()
        response = model.invoke([HumanMessage(content=prompt)])
        end_time = time.time()
        
        print(f"✅ {model_name} responded in {end_time - start_time:.2f}s")
        print(f"Response: {response.content[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ {model_name} failed: {e}")
        return False

def main():
    """Test all available models"""
    models = [
        "phi3:mini",
        "llama3.2:3b", 
        "qwen2:7b",
        "mistral:7b-instruct-v0.3-q8_0"
    ]
    
    print("🧪 Testing Ollama models...")
    
    for model in models:
        success = test_model(model)
        print("-" * 50)
        
        if success:
            print(f"✅ {model} is working!")
            break
    else:
        print("❌ No models are working properly")

if __name__ == "__main__":
    main()
