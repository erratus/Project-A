#!/usr/bin/env python3
"""
Utility functions for embedding-based resume-JD comparison system

This module provides helper functions for managing embeddings, analyzing results,
and performing various operations on the comparison system.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# === Configuration ===
JD_DB_PATH = "chroma_db_jd"
RESUME_DB_PATH = "chroma_db_resume"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class EmbeddingAnalyzer:
    """Utility class for analyzing embeddings and comparison results"""
    
    def __init__(self):
        """Initialize the analyzer"""
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        if os.path.exists(JD_DB_PATH):
            self.jd_store = Chroma(
                persist_directory=JD_DB_PATH, 
                embedding_function=self.embedding_model
            )
        else:
            self.jd_store = None
            
        if os.path.exists(RESUME_DB_PATH):
            self.resume_store = Chroma(
                persist_directory=RESUME_DB_PATH, 
                embedding_function=self.embedding_model
            )
        else:
            self.resume_store = None
    
    def check_embedding_stores(self) -> Dict[str, Any]:
        """Check the status of embedding stores"""
        status = {
            "jd_store": {
                "exists": os.path.exists(JD_DB_PATH),
                "path": JD_DB_PATH,
                "document_count": 0
            },
            "resume_store": {
                "exists": os.path.exists(RESUME_DB_PATH),
                "path": RESUME_DB_PATH,
                "document_count": 0
            }
        }
        
        if self.jd_store:
            try:
                jd_docs = self.jd_store.similarity_search(".", k=1000)
                status["jd_store"]["document_count"] = len(jd_docs)
            except:
                status["jd_store"]["document_count"] = "Error retrieving count"
        
        if self.resume_store:
            try:
                resume_docs = self.resume_store.similarity_search(".", k=1000)
                status["resume_store"]["document_count"] = len(resume_docs)
            except:
                status["resume_store"]["document_count"] = "Error retrieving count"
        
        return status
    
    def analyze_embedding_quality(self, sample_size: int = 5) -> Dict[str, Any]:
        """Analyze the quality and distribution of embeddings"""
        analysis = {
            "embedding_model": EMBEDDING_MODEL,
            "sample_size": sample_size,
            "jd_analysis": {},
            "resume_analysis": {}
        }
        
        # Analyze JD embeddings
        if self.jd_store:
            jd_docs = self.jd_store.similarity_search(".", k=sample_size)
            jd_embeddings = []
            jd_contents = []
            
            for doc in jd_docs:
                embedding = self.embedding_model.embed_query(doc.page_content)
                jd_embeddings.append(embedding)
                jd_contents.append(doc.page_content[:200] + "...")
            
            if jd_embeddings:
                jd_embeddings = np.array(jd_embeddings)
                analysis["jd_analysis"] = {
                    "sample_count": len(jd_embeddings),
                    "embedding_dimension": jd_embeddings.shape[1],
                    "mean_magnitude": float(np.mean(np.linalg.norm(jd_embeddings, axis=1))),
                    "std_magnitude": float(np.std(np.linalg.norm(jd_embeddings, axis=1))),
                    "sample_contents": jd_contents
                }
        
        # Analyze Resume embeddings
        if self.resume_store:
            resume_docs = self.resume_store.similarity_search(".", k=sample_size)
            resume_embeddings = []
            resume_contents = []
            
            for doc in resume_docs:
                embedding = self.embedding_model.embed_query(doc.page_content)
                resume_embeddings.append(embedding)
                resume_contents.append(doc.page_content[:200] + "...")
            
            if resume_embeddings:
                resume_embeddings = np.array(resume_embeddings)
                analysis["resume_analysis"] = {
                    "sample_count": len(resume_embeddings),
                    "embedding_dimension": resume_embeddings.shape[1],
                    "mean_magnitude": float(np.mean(np.linalg.norm(resume_embeddings, axis=1))),
                    "std_magnitude": float(np.std(np.linalg.norm(resume_embeddings, axis=1))),
                    "sample_contents": resume_contents
                }
        
        return analysis
    
    def find_similar_documents(self, query: str, doc_type: str = "both", k: int = 5) -> Dict[str, Any]:
        """Find documents similar to a query"""
        results = {
            "query": query,
            "doc_type": doc_type,
            "results": []
        }
        
        if doc_type in ["jd", "both"] and self.jd_store:
            jd_matches = self.jd_store.similarity_search_with_score(query, k=k)
            for doc, score in jd_matches:
                results["results"].append({
                    "type": "job_description",
                    "source": doc.metadata.get('source', 'unknown'),
                    "similarity_score": float(score),
                    "content_preview": doc.page_content[:300] + "...",
                    "metadata": doc.metadata
                })
        
        if doc_type in ["resume", "both"] and self.resume_store:
            resume_matches = self.resume_store.similarity_search_with_score(query, k=k)
            for doc, score in resume_matches:
                results["results"].append({
                    "type": "resume",
                    "source": doc.metadata.get('source', 'unknown'),
                    "similarity_score": float(score),
                    "content_preview": doc.page_content[:300] + "...",
                    "metadata": doc.metadata
                })
        
        # Sort by similarity score
        results["results"].sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return results
    
    def compare_two_documents(self, doc1_id: str, doc2_id: str) -> Dict[str, Any]:
        """Compare two specific documents by their IDs"""
        # This is a simplified version - in practice, you'd need to implement
        # document retrieval by ID and then compute similarity
        return {
            "doc1_id": doc1_id,
            "doc2_id": doc2_id,
            "note": "Direct document comparison by ID not implemented yet"
        }

def analyze_comparison_results(results_file: str) -> Dict[str, Any]:
    """Analyze results from a comparison run"""
    if not os.path.exists(results_file):
        return {"error": f"Results file not found: {results_file}"}
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    analysis = {
        "file": results_file,
        "comparison_type": results.get("comparison_type", "unknown"),
        "total_matches": results.get("total_matches_above_threshold", 0),
        "processing_time": results.get("processing_time_seconds", 0),
        "statistics": {}
    }
    
    # Analyze top matches
    if "top_matches" in results:
        top_matches = results["top_matches"]
        
        if top_matches:
            embedding_scores = [match["embedding_similarity"] for match in top_matches]
            analysis["statistics"]["embedding_scores"] = {
                "count": len(embedding_scores),
                "mean": np.mean(embedding_scores),
                "std": np.std(embedding_scores),
                "min": np.min(embedding_scores),
                "max": np.max(embedding_scores)
            }
            
            # Analyze detailed analysis scores if available
            overall_scores = []
            for match in top_matches:
                if "detailed_analysis" in match and "overall_match" in match["detailed_analysis"]:
                    if "score" in match["detailed_analysis"]["overall_match"]:
                        overall_scores.append(match["detailed_analysis"]["overall_match"]["score"])
            
            if overall_scores:
                analysis["statistics"]["overall_scores"] = {
                    "count": len(overall_scores),
                    "mean": np.mean(overall_scores),
                    "std": np.std(overall_scores),
                    "min": np.min(overall_scores),
                    "max": np.max(overall_scores)
                }
    
    return analysis

def print_system_status():
    """Print the current status of the embedding system"""
    print("=== Embedding System Status ===")
    
    analyzer = EmbeddingAnalyzer()
    status = analyzer.check_embedding_stores()
    
    print(f"\nJob Description Store:")
    print(f"  Path: {status['jd_store']['path']}")
    print(f"  Exists: {status['jd_store']['exists']}")
    print(f"  Documents: {status['jd_store']['document_count']}")
    
    print(f"\nResume Store:")
    print(f"  Path: {status['resume_store']['path']}")
    print(f"  Exists: {status['resume_store']['exists']}")
    print(f"  Documents: {status['resume_store']['document_count']}")
    
    print(f"\nEmbedding Model: {EMBEDDING_MODEL}")
    
    # Check if comparison results exist
    output_dir = "output"
    if os.path.exists(output_dir):
        result_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
        print(f"\nPrevious Results: {len(result_files)} files found in {output_dir}/")
        for f in result_files[-3:]:  # Show last 3 files
            print(f"  - {f}")
    else:
        print(f"\nNo output directory found")

def main():
    """Main function for utility operations"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Embedding System Utilities")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--analyze-quality", action="store_true", help="Analyze embedding quality")
    parser.add_argument("--search", type=str, help="Search for similar documents")
    parser.add_argument("--doc-type", choices=["jd", "resume", "both"], default="both",
                       help="Type of documents to search")
    parser.add_argument("--analyze-results", type=str, help="Analyze comparison results file")
    
    args = parser.parse_args()
    
    if args.status:
        print_system_status()
    
    elif args.analyze_quality:
        analyzer = EmbeddingAnalyzer()
        quality_analysis = analyzer.analyze_embedding_quality()
        print("\n=== Embedding Quality Analysis ===")
        print(json.dumps(quality_analysis, indent=2))
    
    elif args.search:
        analyzer = EmbeddingAnalyzer()
        search_results = analyzer.find_similar_documents(args.search, args.doc_type)
        print(f"\n=== Search Results for '{args.search}' ===")
        for i, result in enumerate(search_results["results"][:5]):
            print(f"\n{i+1}. [{result['type'].upper()}] {result['source']}")
            print(f"   Similarity: {result['similarity_score']:.4f}")
            print(f"   Preview: {result['content_preview'][:150]}...")
    
    elif args.analyze_results:
        analysis = analyze_comparison_results(args.analyze_results)
        print(f"\n=== Analysis of {args.analyze_results} ===")
        print(json.dumps(analysis, indent=2))
    
    else:
        print("Use --help to see available options")

if __name__ == "__main__":
    main()
