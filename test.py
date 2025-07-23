import requests
import json

def query_ollama(prompt: str, model: str = "bigllama/mistralv01-7b:latest", temperature: float = 0.0, seed: int = 42):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # Disable streaming for simpler handling
        "options": {
            "temperature": temperature,
            "seed": seed
        }
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["response"].strip()
    else:
        raise RuntimeError(f"Failed to generate response: {response.text}")

# Alternative function that handles streaming responses
def query_ollama_streaming(prompt: str, model: str = "bigllama/mistralv01-7b:latest", temperature: float = 0.0, seed: int = 42):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "seed": seed
        }
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        # Handle streaming response - each line is a separate JSON object
        full_response = ""
        for line in response.text.strip().split('\n'):
            if line.strip():
                try:
                    json_response = json.loads(line)
                    if 'response' in json_response:
                        full_response += json_response['response']
                    # Check if this is the final response
                    if json_response.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
        return full_response.strip()
    else:
        raise RuntimeError(f"Failed to generate response: {response.text}")

if __name__ == "__main__":
    # Example prompt
    prompt = """You are a job matching expert. Given the resume and job description below, return a match percentage.

Resume:
- Skills: Python, Data Analysis, Machine Learning
- Experience: 2 years as Data Scientist
- Education: B.Tech in Computer Science

Job Description:
- Skills: Python, Deep Learning, SQL
- Experience: 2+ years in AI/ML
- Education: B.Tech in CS or related

Return only the match percentage and explain briefly."""

    try:
        # Use the non-streaming version (recommended)
        result = query_ollama(prompt)
        print("Model Output:")
        print(result)
    except Exception as e:
        print(f"Error with non-streaming: {e}")
        
        # Fallback to streaming version
        try:
            result = query_ollama_streaming(prompt)
            print("Model Output (streaming):")
            print(result)
        except Exception as e2:
            print(f"Error with streaming: {e2}")