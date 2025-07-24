#!/usr/bin/env python3
"""
Simple Embedding-based Resume-JD Comparison Script

This is a lightweight version that focuses on embedding similarity without LLM analysis.
Perfect for quick comparisons and when LLM is not available.
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Tuple, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sklearn.metrics.pairwise import cosine_similarity

# === Configuration ===
JD_DB_PATH = "chroma_db_jd"
RESUME_DB_PATH = "chroma_db_resume"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "output"

class SimpleEmbeddingComparator:
    """Lightweight embedding-based comparator without LLM dependency"""
    
    def __init__(self, similarity_threshold=0.6):
        """Initialize the comparator"""
        print("[INFO] Initializing Simple Embedding Comparator...")
        
        self.similarity_threshold = similarity_threshold
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize vector stores
        self.jd_store = Chroma(
            persist_directory=JD_DB_PATH, 
            embedding_function=self.embedding_model
        )
        self.resume_store = Chroma(
            persist_directory=RESUME_DB_PATH, 
            embedding_function=self.embedding_model
        )
        
        print("[INFO] Initialization complete!")
    
    def get_all_documents(self, store: Chroma, store_name: str) -> List[Dict]:
        """Retrieve all documents from a vector store"""
        try:
            docs = store.similarity_search(".", k=1000)
            
            documents = []
            for doc in docs:
                documents.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'id': doc.metadata.get('source', 'unknown')
                })
            
            print(f"[INFO] Retrieved {len(documents)} documents from {store_name}")
            return documents
            
        except Exception as e:
            print(f"[ERROR] Failed to retrieve documents from {store_name}: {e}")
            return []
    
    def compute_all_similarities(self) -> List[Dict]:
        """Compute similarities between all resume-JD pairs"""
        print("[INFO] Computing embedding similarities...")
        
        # Get all documents
        resume_docs = self.get_all_documents(self.resume_store, "Resume Store")
        jd_docs = self.get_all_documents(self.jd_store, "JD Store")
        
        if not resume_docs or not jd_docs:
            print("[ERROR] No documents found in one or both stores!")
            return []
        
        matches = []
        
        for resume_doc in resume_docs:
            resume_embedding = self.embedding_model.embed_query(resume_doc['content'])
            
            for jd_doc in jd_docs:
                jd_embedding = self.embedding_model.embed_query(jd_doc['content'])
                
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    [resume_embedding], 
                    [jd_embedding]
                )[0][0]
                
                # Only include matches above threshold
                if similarity >= self.similarity_threshold:
                    match = {
                        'resume_id': resume_doc['id'],
                        'jd_id': jd_doc['id'],
                        'similarity_score': float(similarity),
                        'resume_content': resume_doc['content'],
                        'jd_content': jd_doc['content'],
                        'resume_metadata': resume_doc['metadata'],
                        'jd_metadata': jd_doc['metadata'],
                        'analysis': self._create_simple_analysis(similarity, resume_doc, jd_doc)
                    }
                    matches.append(match)
        
        # Sort matches by similarity score (descending)
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        print(f"[INFO] Found {len(matches)} matches above threshold ({self.similarity_threshold})")
        return matches
    
    def _create_simple_analysis(self, similarity: float, resume_doc: Dict, jd_doc: Dict) -> Dict:
        """Create a simple analysis based on embedding similarity"""
        base_score = min(100, similarity * 100)
        
        # Extract key information from content
        resume_content = resume_doc['content'].lower()
        jd_content = jd_doc['content'].lower()
        
        # Simple keyword overlap analysis
        resume_skills = self._extract_skills(resume_content)
        jd_skills = self._extract_skills(jd_content)
        
        skill_overlap = len(resume_skills.intersection(jd_skills))
        skill_total = len(jd_skills) if jd_skills else 1
        skill_match_pct = min(100, (skill_overlap / skill_total) * 100)
        
        return {
            "embedding_similarity": similarity,
            "similarity_percentage": base_score,
            "skill_overlap_count": skill_overlap,
            "skill_match_percentage": skill_match_pct,
            "resume_skills_found": list(resume_skills)[:10],  # Top 10
            "jd_skills_required": list(jd_skills)[:10],  # Top 10
            "recommendation": self._get_recommendation(similarity),
            "analysis_method": "embedding_based_simple"
        }
    
    def _extract_skills(self, content: str) -> set:
        """Extract potential skills from content"""
        # Common technical skills and keywords
        skills_keywords = {
            'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'sql', 'nosql',
            'machine learning', 'deep learning', 'nlp', 'computer vision',
            'generative ai', 'llm', 'transformer', 'bert', 'gpt', 'langchain',
            'rag', 'vector database', 'embedding', 'fine-tuning', 'prompt engineering',
            'devops', 'mlops', 'ci/cd', 'jenkins', 'terraform', 'ansible'
        }
        
        found_skills = set()
        for skill in skills_keywords:
            if skill in content:
                found_skills.add(skill)
        
        return found_skills
    
    def _get_recommendation(self, similarity: float) -> str:
        """Get recommendation based on similarity score"""
        if similarity >= 0.85:
            return "strong_match - excellent candidate"
        elif similarity >= 0.75:
            return "good_match - strong candidate"
        elif similarity >= 0.65:
            return "moderate_match - consider for interview"
        else:
            return "weak_match - review carefully"
    
    def run_comparison(self) -> Dict[str, Any]:
        """Run the complete comparison"""
        print("\n" + "="*60)
        print("STARTING SIMPLE EMBEDDING-BASED COMPARISON")
        print("="*60)
        
        start_time = time.time()
        
        # Compute all similarities
        matches = self.compute_all_similarities()
        
        if not matches:
            return {
                "error": "No matches found above threshold",
                "threshold": self.similarity_threshold
            }
        
        # Compile results
        results = {
            "comparison_type": "simple_embedding_based",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_seconds": round(time.time() - start_time, 2),
            "similarity_threshold": self.similarity_threshold,
            "embedding_model": EMBEDDING_MODEL,
            "total_matches": len(matches),
            "matches": matches,
            "summary": {
                "top_similarity": matches[0]['similarity_score'] if matches else 0,
                "average_similarity": np.mean([m['similarity_score'] for m in matches]),
                "recommendations": {
                    "strong_match": len([m for m in matches if m['similarity_score'] >= 0.85]),
                    "good_match": len([m for m in matches if 0.75 <= m['similarity_score'] < 0.85]),
                    "moderate_match": len([m for m in matches if 0.65 <= m['similarity_score'] < 0.75]),
                    "weak_match": len([m for m in matches if m['similarity_score'] < 0.65])
                }
            }
        }
        
        print(f"\n[SUCCESS] Comparison completed in {results['processing_time_seconds']} seconds")
        return results

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Embedding-based Resume-JD Comparison")
    parser.add_argument("--threshold", type=float, default=0.6,
                       help="Similarity threshold for matches")
    parser.add_argument("--output", type=str, default="simple_embedding_results.json",
                       help="Output filename")
    
    args = parser.parse_args()
    
    print("Starting Simple Embedding-Based Resume-JD Comparison")
    print("-" * 50)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize comparator
    comparator = SimpleEmbeddingComparator(similarity_threshold=args.threshold)
    
    # Run comparison
    results = comparator.run_comparison()
    
    # Save results
    output_file = os.path.join(OUTPUT_DIR, args.output)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SUCCESS] Results saved to: {output_file}")
    
    # Print summary
    if "error" not in results:
        print(f"\nSUMMARY:")
        print(f"- Total matches found: {results['total_matches']}")
        print(f"- Processing time: {results['processing_time_seconds']} seconds")
        print(f"- Average similarity: {results['summary']['average_similarity']:.4f}")
        
        print(f"\nRECOMMENDATION BREAKDOWN:")
        for rec_type, count in results['summary']['recommendations'].items():
            print(f"- {rec_type.replace('_', ' ').title()}: {count}")
        
        print(f"\nTOP 3 MATCHES:")
        for i, match in enumerate(results['matches'][:3]):
            print(f"{i+1}. {match['resume_id']} vs {match['jd_id']}")
            print(f"   Similarity: {match['similarity_score']:.4f}")
            print(f"   Recommendation: {match['analysis']['recommendation']}")
            print(f"   Skill overlap: {match['analysis']['skill_overlap_count']} skills")
            print()

if __name__ == "__main__":
    main()
