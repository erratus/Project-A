import asyncio
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import multiprocessing as mp
from dataclasses import dataclass
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

@dataclass
class OptimizationConfig:
    """Configuration for SLM optimization"""
    # Model selection - choose fastest SLM for your hardware
    model_name: str = "phi3:mini"  # or "qwen2:7b", "mistral:7b-instruct-v0.3"
    
    # Concurrency settings
    max_workers: int = min(4, mp.cpu_count())  # Adjust based on your hardware
    batch_size: int = 10  # Process resumes in batches
    
    # Model settings for speed
    temperature: float = 0.0  # Deterministic, faster inference
    max_tokens: int = 1024    # Limit output length
    timeout: int = 30         # Timeout per request
    
    # Caching
    enable_caching: bool = True
    cache_dir: str = "model_cache"

class OptimizedSLMComparator:
    """Optimized resume comparison using Small Language Models"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.cache = {}
        
        # Initialize multiple model instances for parallel processing
        self.models = []
        for i in range(config.max_workers):
            model = ChatOllama(
                model=config.model_name,
                temperature=config.temperature,
                seed=42 + i,  # Different seeds for parallel instances
                timeout=config.timeout,
                num_ctx=2048,  # Reduced context window for speed
                num_predict=config.max_tokens,
                # Optimization parameters for speed
                num_thread=2,  # Threads per model instance
                repeat_penalty=1.0,
                top_k=10,      # Reduced for faster sampling
                top_p=0.9
            )
            self.models.append(model)
    
    def create_optimized_prompt(self, resume_data: dict, jd_data: dict, resume_filename: str) -> str:
        """Create a more concise, focused prompt for faster processing"""
        
        # Simplified system prompt focusing on key comparisons
        system_prompt = """You are an expert resume-JD matcher. Analyze and return ONLY this JSON structure:

{
  "resume_filename": {
    "Skills": {"match_pct": float, "explanation": "brief"},
    "Education": {"match_pct": float, "explanation": "brief"}, 
    "Job Role": {"match_pct": float, "explanation": "brief"},
    "Experience": {"match_pct": float, "explanation": "brief"},
    "OverallMatchPercentage": float,
    "why_overall_match_is_this": "brief explanation"
  }
}

Rules:
- match_pct: 0-100 scale
- Focus on semantic similarity
- Keep explanations under 20 words
- Strong domain matches get 80%+ scores
- Return only valid JSON"""

        # Simplified data formatting
        resume_skills = ', '.join(resume_data.get('skill', []))[:200]
        resume_education = ', '.join(resume_data.get('education', []))[:100]
        resume_experience = ', '.join(resume_data.get('experience', []))[:300]
        resume_role = ', '.join(resume_data.get('job role', []))[:50]
        
        jd_skills = ', '.join(jd_data.get('skill', []))[:200]
        jd_education = ', '.join(jd_data.get('education', []))[:100]
        jd_experience = ', '.join(jd_data.get('experience', []))[:300]
        jd_role = ', '.join(jd_data.get('job role', []))[:50]
        
        user_prompt = f"""Compare resume vs job requirements:

RESUME ({resume_filename}):
Skills: {resume_skills}
Education: {resume_education}
Role: {resume_role}
Experience: {resume_experience}

JOB REQUIREMENTS:
Skills: {jd_skills}
Education: {jd_education}
Role: {jd_role}
Experience: {jd_experience}

Return JSON for: {resume_filename}"""
        
        return system_prompt + "\n\n" + user_prompt
    
    def process_single_comparison(self, resume_data: dict, jd_data: dict, 
                                 resume_filename: str, model_idx: int) -> dict:
        """Process a single resume comparison with caching"""
        
        # Check cache first
        cache_key = f"{resume_filename}_{hash(str(jd_data))}"
        if self.config.enable_caching and cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            start_time = time.time()
            
            # Create optimized prompt
            prompt = self.create_optimized_prompt(resume_data, jd_data, resume_filename)
            
            # Use specific model instance for this comparison
            model = self.models[model_idx % len(self.models)]
            
            # Single message approach (faster than system + human message split)
            response = model.invoke([HumanMessage(content=prompt)])
            
            # Quick JSON extraction
            content = response.content.strip()
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            
            # Find JSON boundaries
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_content = content[json_start:json_end]
                result = json.loads(json_content)
                
                # Cache the result
                if self.config.enable_caching:
                    self.cache[cache_key] = result
                
                processing_time = time.time() - start_time
                print(f"✅ {resume_filename}: {processing_time:.2f}s")
                
                return result
            else:
                raise ValueError("No valid JSON found")
                
        except Exception as e:
            print(f"❌ Error processing {resume_filename}: {e}")
            return {resume_filename: {"error": str(e)}}
    
    def process_batch_parallel(self, resume_batch: List[dict], jd_data: dict) -> List[dict]:
        """Process a batch of resumes in parallel"""
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all comparisons in the batch
            future_to_resume = {}
            
            for i, resume_data in enumerate(resume_batch):
                resume_filename = resume_data.get('filename', f'resume_{i}.json')
                
                future = executor.submit(
                    self.process_single_comparison,
                    resume_data, jd_data, resume_filename, i
                )
                future_to_resume[future] = resume_filename
            
            # Collect results as they complete
            for future in as_completed(future_to_resume):
                resume_filename = future_to_resume[future]
                try:
                    result = future.result(timeout=self.config.timeout)
                    results.append(result)
                except Exception as e:
                    print(f"❌ Batch error for {resume_filename}: {e}")
                    results.append({resume_filename: {"error": str(e)}})
        
        return results
    
    def run_optimized_comparison(self, resumes: List[dict], jd_data: dict) -> dict:
        """Run optimized comparison with batching and parallelization"""
        
        print(f"🚀 Starting optimized comparison with {self.config.model_name}")
        print(f"📊 Processing {len(resumes)} resumes in batches of {self.config.batch_size}")
        print(f"⚡ Using {self.config.max_workers} parallel workers")
        
        start_time = time.time()
        all_results = {}
        
        # Process in batches
        for i in range(0, len(resumes), self.config.batch_size):
            batch = resumes[i:i + self.config.batch_size]
            batch_num = (i // self.config.batch_size) + 1
            total_batches = (len(resumes) + self.config.batch_size - 1) // self.config.batch_size
            
            print(f"\n📦 Processing batch {batch_num}/{total_batches} ({len(batch)} resumes)")
            
            batch_start = time.time()
            batch_results = self.process_batch_parallel(batch, jd_data)
            batch_time = time.time() - batch_start
            
            # Merge batch results
            for result in batch_results:
                all_results.update(result)
            
            print(f"✅ Batch {batch_num} completed in {batch_time:.2f}s")
        
        total_time = time.time() - start_time
        avg_time_per_resume = total_time / len(resumes)
        
        print(f"\n🎯 OPTIMIZATION RESULTS:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average per resume: {avg_time_per_resume:.2f}s")
        print(f"   Throughput: {len(resumes)/total_time:.2f} resumes/second")
        
        return all_results

# Performance comparison function
def compare_model_performance():
    """Benchmark different SLMs for your specific use case"""
    
    models_to_test = [
        "phi3:mini",           # 3.8B - Very fast
        "qwen2:7b",           # 7B - Good balance
        "mistral:7b-instruct-v0.3-q8_0 ",  # Your current model optimized
        "llama3.2:3b",        # Alternative 3B model
    ]
    
    # Sample data for benchmarking
    sample_resume = {
        "skill": ["Python", "Machine Learning", "TensorFlow"],
        "education": ["B.Tech Computer Science"],
        "job role": ["Data Scientist"],
        "experience": ["3 years ML development"],
        "filename": "test_resume.json"
    }
    
    sample_jd = {
        "skill": ["Python", "AI", "Deep Learning"],
        "education": ["Bachelor's degree"],
        "job role": ["ML Engineer"],
        "experience": ["2+ years experience"]
    }
    
    print("🏃‍♂️ PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    for model_name in models_to_test:
        try:
            config = OptimizationConfig(model_name=model_name, max_workers=1)
            comparator = OptimizedSLMComparator(config)
            
            # Warmup run
            comparator.process_single_comparison(sample_resume, sample_jd, "warmup.json", 0)
            
            # Benchmark run
            start_time = time.time()
            result = comparator.process_single_comparison(sample_resume, sample_jd, "benchmark.json", 0)
            end_time = time.time()
            
            success = "error" not in result.get("benchmark.json", {})
            
            print(f"📊 {model_name}:")
            print(f"   Time: {end_time - start_time:.3f}s")
            print(f"   Success: {'✅' if success else '❌'}")
            print()
            
        except Exception as e:
            print(f"❌ {model_name}: Failed - {e}")

# Usage example
def main():
    """Example usage of optimized SLM comparison"""
    
    # Run performance benchmark first
    compare_model_performance()
    
    # Configure for your best performing model
    config = OptimizationConfig(
        model_name="phi3:mini",  # Replace with your fastest model
        max_workers=4,
        batch_size=8,
        enable_caching=True
    )
    
    # Initialize comparator
    comparator = OptimizedSLMComparator(config)
    
    # Load your data (placeholder - replace with your actual data loading)
    resumes = []  # Load from your resume_json directory
    jd_data = {}  # Load from your JD_extraction directory
    
    # Run optimized comparison
    results = comparator.run_optimized_comparison(resumes, jd_data)
    
    # Save results
    os.makedirs("output", exist_ok=True)
    with open("output/slm_optimized_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("✅ Optimized comparison completed!")

if __name__ == "__main__":
    main()