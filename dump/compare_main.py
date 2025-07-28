#!/usr/bin/env python3
"""
Embedding-based Resume-JD Comparison Script

This script performs semantic comparison between resumes and job descriptions
using embeddings instead of JSON extractions. It leverages ChromaDB vector
similarity search to find the most relevant matches and provides detailed
comparison results.
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from sklearn.metrics.pairwise import cosine_similarity

# === Configuration ===
JD_DB_PATH = "chroma_db_jd"
RESUME_DB_PATH = "chroma_db_resume"
MODEL_NAME = "bigllama/mistralv01-7b:latest"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "output"
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score for matches

@dataclass
class EmbeddingMatch:
    """Data class to store embedding match results"""
    resume_id: str
    jd_id: str
    similarity_score: float
    resume_content: str
    jd_content: str
    resume_metadata: Dict
    jd_metadata: Dict

class EmbeddingComparator:
    """Main class for embedding-based resume-JD comparison"""

    def __init__(self, similarity_threshold=SIMILARITY_THRESHOLD):
        """Initialize the comparator with embedding models and vector stores"""
        print("[INFO] Initializing Embedding Comparator...")

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
        
        # Initialize LLM for detailed analysis
        self.chat = ChatOllama(model=MODEL_NAME, temperature=0.0, seed=42)
        
        print("[INFO] Initialization complete!")
    
    def get_all_documents(self, store: Chroma, store_name: str) -> List[Dict]:
        """Retrieve all documents from a vector store"""
        try:
            # Get all documents using similarity search with a generic query
            docs = store.similarity_search(".", k=1000)  # Large k to get all docs
            
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
    
    def compute_embedding_similarity(self, resume_docs: List[Dict], jd_docs: List[Dict]) -> List[EmbeddingMatch]:
        """Compute similarity between resume and JD embeddings"""
        print("[INFO] Computing embedding similarities...")
        
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
                    match = EmbeddingMatch(
                        resume_id=resume_doc['id'],
                        jd_id=jd_doc['id'],
                        similarity_score=float(similarity),
                        resume_content=resume_doc['content'],
                        jd_content=jd_doc['content'],
                        resume_metadata=resume_doc['metadata'],
                        jd_metadata=jd_doc['metadata']
                    )
                    matches.append(match)
        
        # Sort matches by similarity score (descending)
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        
        print(f"[INFO] Found {len(matches)} matches above threshold ({self.similarity_threshold})")
        return matches
    
    def find_best_matches(self, resume_query: str, jd_query: str, k: int = 5) -> List[EmbeddingMatch]:
        """Find best matches using vector similarity search"""
        print(f"[INFO] Finding best matches for queries...")
        
        # Search for similar resumes and JDs
        similar_resumes = self.resume_store.similarity_search_with_score(resume_query, k=k)
        similar_jds = self.jd_store.similarity_search_with_score(jd_query, k=k)
        
        matches = []
        
        for resume_doc, resume_score in similar_resumes:
            for jd_doc, jd_score in similar_jds:
                # Calculate combined similarity score
                combined_score = (resume_score + jd_score) / 2
                
                match = EmbeddingMatch(
                    resume_id=resume_doc.metadata.get('source', 'unknown'),
                    jd_id=jd_doc.metadata.get('source', 'unknown'),
                    similarity_score=float(combined_score),
                    resume_content=resume_doc.page_content,
                    jd_content=jd_doc.page_content,
                    resume_metadata=resume_doc.metadata,
                    jd_metadata=jd_doc.metadata
                )
                matches.append(match)
        
        # Sort by similarity score
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return matches
    
    def generate_detailed_analysis(self, match: EmbeddingMatch) -> Dict[str, Any]:
        """Generate detailed analysis using LLM for a specific match"""
        
        system_prompt = """You are a world-class HR, Talent Acquisition, and Generative AI Specialist with deep expertise in job-role alignment, semantic document comparison, and hiring decision automation.

You are tasked with comparing a candidate resume and a job description. Your job is to assess the alignment **strictly based on meaning** — not exact keyword matches.

You must return a single valid JSON object in the structure described below.

### Instructions:

- Evaluate **semantic relevance**, not just keyword overlap. For example, treat "ML Engineer" and "Machine Learning Engineer" as identical.
- Use **real-world hiring logic**: If a resume **exceeds** the JD requirements (e.g., more skills, more education, deeper experience), the match_pct should be high — even **100%**.
- Avoid over-penalizing minor differences. Focus on **capability and fit**.
- NEVER hallucinate or infer information not explicitly present in either document.
- NEVER nest objects inside any field — your output must remain a **flat JSON**.
- Explanations must be **insightful, human-readable, and professional** — written as if speaking to a hiring manager.
- Do **not** include any commentary or text outside the JSON.

### Field Matching Logic:

1. **Skills**
   - Match based on technical equivalence and semantic similarity.
   - If the resume includes **all** required skills or **more**, assign **85-100%**.
   - If semantically similar (e.g., "Deep Learning" covers "TensorFlow/PyTorch"), assign high match_pct (80–95%).
   - Consider domain expertise: "Generative AI" + "LLMs" is highly relevant for "ML Engineer/Generative AI Engineer".

2. **Education**
   - If the candidate's education level is **equal or higher** than the JD, score high (90-100%).
   - Post-graduate degrees in relevant fields should score very high.
   - B.Tech in CS/IT perfectly matches requirements.

3. **Experience**
   - Match on role relevance, technologies used, domain familiarity, and years of experience.
   - Experience that directly meets or exceeds JD expectations should score high (85–100%).
   - Hands-on ML/AI experience is highly valuable.

4. **Job Role**
   - "Data Scientist" with ML/AI focus is highly compatible with "ML Engineer/Generative AI Engineer" (85-95%).
   - Focus on domain alignment and technical overlap.

5. **OverallMatchPercentage**
   - Must be a weighted score calculated from Skills, Experience, Education, and Job Role.
   - Strong candidates should score 85-95%.

### Output Format (strict):

{
  "Skills": {
    "match_pct": float,
    "resume_value": string,
    "job_description_value": string,
    "explanation": string
  },
  "Education": {
    "match_pct": float,
    "resume_value": string,
    "job_description_value": string,
    "explanation": string
  },
  "Job Role": {
    "match_pct": float,
    "resume_value": string,
    "job_description_value": string,
    "explanation": string
  },
  "Experience": {
    "match_pct": float,
    "resume_value": string,
    "job_description_value": string,
    "explanation": string
  },
  "OverallMatchPercentage": float,
  "why_overall_match_is_this": string,
  "AI_Generated_Estimate_Percentage": float
}

Return ONLY the JSON object. No extra comments or explanation."""

        user_prompt = f"""
EMBEDDING SIMILARITY SCORE: {match.similarity_score:.4f}

RESUME CONTENT:
{match.resume_content}

JOB DESCRIPTION CONTENT:
{match.jd_content}

RESUME METADATA: {json.dumps(match.resume_metadata, indent=2)}
JD METADATA: {json.dumps(match.jd_metadata, indent=2)}

Please provide a detailed semantic analysis of this match."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            print(f"[DEBUG] Sending request to LLM for {match.resume_id} vs {match.jd_id}")
            response = self.chat.invoke(messages)
            content = response.content.strip()

            print(f"[DEBUG] LLM response length: {len(content)} characters")

            if not content:
                print(f"[WARNING] Empty response from LLM for {match.resume_id} vs {match.jd_id}")
                return self._create_fallback_analysis(match)

            # Clean up response if needed
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            elif content.startswith('```'):
                content = content.replace('```', '').strip()

            # Try to parse JSON
            try:
                analysis = json.loads(content)
                analysis['embedding_similarity'] = match.similarity_score
                return analysis
            except json.JSONDecodeError as json_error:
                print(f"[WARNING] JSON parsing failed for {match.resume_id} vs {match.jd_id}: {json_error}")
                print(f"[DEBUG] Raw content: {content[:200]}...")
                return self._create_fallback_analysis(match, llm_response=content[:500])

        except Exception as e:
            print(f"[ERROR] Failed to generate analysis for {match.resume_id} vs {match.jd_id}: {e}")
            return self._create_fallback_analysis(match, error=str(e))

    def _create_fallback_analysis(self, match: EmbeddingMatch, error: str = None, llm_response: str = None) -> Dict[str, Any]:
        """Create a fallback analysis when LLM fails - similar to compare_direct.py format"""
        # Calculate basic scores based on embedding similarity
        base_score = min(100, match.similarity_score * 100)

        # Extract content for analysis
        resume_content = match.resume_content
        jd_content = match.jd_content

        # Simple skill extraction
        resume_skills = self._extract_skills_from_content(resume_content)
        jd_skills = self._extract_skills_from_content(jd_content)

        # Calculate skill overlap with better scoring
        skill_overlap = len(resume_skills.intersection(jd_skills))
        skill_total = len(jd_skills) if jd_skills else 1

        # More generous scoring for skill matches
        if skill_overlap >= skill_total * 0.8:  # 80%+ overlap
            skill_match_pct = 90 + (skill_overlap / skill_total) * 10  # 90-100%
        elif skill_overlap >= skill_total * 0.6:  # 60%+ overlap
            skill_match_pct = 80 + (skill_overlap / skill_total) * 10  # 80-90%
        elif skill_overlap >= skill_total * 0.4:  # 40%+ overlap
            skill_match_pct = 70 + (skill_overlap / skill_total) * 10  # 70-80%
        else:
            skill_match_pct = min(70, (skill_overlap / skill_total) * 100)

        # Bonus for high-value skills
        high_value_skills = {'generative ai', 'llm', 'machine learning', 'deep learning', 'python', 'tensorflow', 'pytorch'}
        resume_high_value = resume_skills.intersection(high_value_skills)
        if len(resume_high_value) >= 3:
            skill_match_pct = min(100, skill_match_pct + 10)  # Bonus for multiple high-value skills

        return {
            "Skills": {
                "match_pct": skill_match_pct / 100,
                "resume_value": ", ".join(list(resume_skills)[:20]),  # Top 20 skills
                "job_description_value": ", ".join(list(jd_skills)[:20]),
                "explanation": f"Based on embedding analysis, found {skill_overlap} overlapping skills out of {len(jd_skills)} required skills. Embedding similarity: {match.similarity_score:.4f}"
            },
            "Education": {
                "match_pct": self._calculate_education_score(resume_content, jd_content, base_score),
                "resume_value": self._extract_education(resume_content),
                "job_description_value": self._extract_education(jd_content),
                "explanation": f"Education scoring based on content analysis. B.Tech/M.Tech in CS/IT or related fields score highly. Post-graduate degrees receive bonus points."
            },
            "Job Role": {
                "match_pct": self._calculate_job_role_score(resume_content, jd_content, base_score),
                "resume_value": self._extract_job_role(resume_content),
                "job_description_value": self._extract_job_role(jd_content),
                "explanation": f"Job role compatibility based on semantic analysis. Data Scientist with ML/AI focus aligns well with ML Engineer/Generative AI Engineer roles."
            },
            "Experience": {
                "match_pct": self._calculate_experience_score(resume_content, jd_content, base_score),
                "resume_value": self._extract_experience(resume_content),
                "job_description_value": self._extract_experience(jd_content),
                "explanation": f"Experience scoring based on years, domain relevance, and technical alignment. Hands-on ML/AI experience receives high scores."
            },
            "OverallMatchPercentage": (base_score * 0.9) / 100,
            "why_overall_match_is_this": f"Overall match of {(base_score * 0.9):.1f}% based on embedding similarity ({match.similarity_score:.4f}). " +
                                       f"Found {skill_overlap} skill overlaps. " +
                                       ("LLM analysis failed: " + str(error) if error else "LLM analysis unavailable."),
            "AI_Generated_Estimate_Percentage": 0.0,  # Cannot estimate without LLM
            "embedding_similarity": match.similarity_score,
            "analysis_method": "fallback_embedding_based",
            "llm_error": error
        }

    def _extract_skills_from_content(self, content: str) -> set:
        """Extract skills from content"""
        content_lower = content.lower()
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
            if skill in content_lower:
                found_skills.add(skill)
        return found_skills

    def _extract_education(self, content: str) -> str:
        """Extract education information from content"""
        lines = content.split('\n')
        for line in lines:
            if 'education:' in line.lower():
                return line.split(':', 1)[1].strip() if ':' in line else line.strip()
        return "Education information not clearly identified"

    def _extract_job_role(self, content: str) -> str:
        """Extract job role from content"""
        lines = content.split('\n')
        for line in lines:
            if 'job role:' in line.lower():
                return line.split(':', 1)[1].strip() if ':' in line else line.strip()
        return "Job role not clearly identified"

    def _extract_experience(self, content: str) -> str:
        """Extract experience information from content"""
        lines = content.split('\n')
        for line in lines:
            if 'experience:' in line.lower():
                return line.split(':', 1)[1].strip() if ':' in line else line.strip()
        return "Experience information not clearly identified"

    def _calculate_education_score(self, resume_content: str, jd_content: str, base_score: float) -> float:
        """Calculate education score with better logic"""
        resume_edu = resume_content.lower()

        # High scores for relevant degrees
        if any(term in resume_edu for term in ['b.tech', 'b tech', 'bachelor', 'computer science', 'computer engineering']):
            score = 90.0
        elif any(term in resume_edu for term in ['m.tech', 'm tech', 'master', 'msc', 'ms']):
            score = 95.0
        else:
            score = base_score * 0.7

        # Bonus for post-graduate degrees in relevant fields
        if any(term in resume_edu for term in ['post graduate', 'diploma', 'data science', 'machine learning', 'ai']):
            score = min(100.0, score + 10.0)

        return score / 100.0

    def _calculate_job_role_score(self, resume_content: str, jd_content: str, base_score: float) -> float:
        """Calculate job role score with better logic"""
        resume_role = resume_content.lower()

        # High compatibility roles
        if any(term in resume_role for term in ['data scientist', 'ml engineer', 'machine learning engineer']):
            score = 90.0
        elif any(term in resume_role for term in ['ai engineer', 'generative ai', 'deep learning']):
            score = 95.0
        elif any(term in resume_role for term in ['software engineer', 'developer', 'analyst']):
            score = 75.0
        else:
            score = base_score * 0.8

        return min(100.0, score) / 100.0

    def _calculate_experience_score(self, resume_content: str, jd_content: str, base_score: float) -> float:
        """Calculate experience score with better logic"""
        resume_exp = resume_content.lower()

        # Look for years of experience
        years_score = 70.0  # Default
        if any(term in resume_exp for term in ['3 years', '4 years', '5 years', '6 years']):
            years_score = 90.0
        elif any(term in resume_exp for term in ['2 years', '7 years', '8 years']):
            years_score = 85.0

        # Look for relevant experience
        relevance_score = 70.0  # Default
        if any(term in resume_exp for term in ['generative ai', 'llm', 'machine learning', 'deep learning']):
            relevance_score = 95.0
        elif any(term in resume_exp for term in ['data science', 'analytics', 'nlp', 'python']):
            relevance_score = 85.0

        # Combined score
        final_score = (years_score + relevance_score) / 2
        return min(100.0, final_score) / 100.0
    
    def run_comprehensive_comparison(self) -> Dict[str, Any]:
        """Run comprehensive embedding-based comparison"""
        print("\n" + "="*60)
        print("STARTING EMBEDDING-BASED RESUME-JD COMPARISON")
        print("="*60)
        
        start_time = time.time()
        
        # Get all documents
        resume_docs = self.get_all_documents(self.resume_store, "Resume Store")
        jd_docs = self.get_all_documents(self.jd_store, "JD Store")
        
        if not resume_docs or not jd_docs:
            print("[ERROR] No documents found in one or both stores!")
            return {"error": "No documents found"}
        
        # Compute all pairwise similarities
        all_matches = self.compute_embedding_similarity(resume_docs, jd_docs)
        
        if not all_matches:
            print(f"[WARNING] No matches found above threshold {self.similarity_threshold}")
            return {"warning": "No matches above threshold", "threshold": self.similarity_threshold}
        
        # Generate detailed analysis for ALL matches
        print(f"[INFO] Generating detailed analysis for all {len(all_matches)} matches...")

        detailed_results = []
        for i, match in enumerate(all_matches):  # Analyze ALL matches
            print(f"[INFO] Analyzing match {i+1}/{len(all_matches)}: {match.resume_id} vs {match.jd_id}")

            analysis = self.generate_detailed_analysis(match)

            result = {
                "rank": i + 1,
                "resume_id": match.resume_id,
                "jd_id": match.jd_id,
                "embedding_similarity": match.similarity_score,
                "detailed_analysis": analysis,
                "resume_metadata": match.resume_metadata,
                "jd_metadata": match.jd_metadata
            }

            detailed_results.append(result)
        
        # Compile final results
        final_results = {
            "comparison_type": "embedding_based",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_seconds": round(time.time() - start_time, 2),
            "total_resumes": len(resume_docs),
            "total_jds": len(jd_docs),
            "total_matches_above_threshold": len(all_matches),
            "similarity_threshold": self.similarity_threshold,
            "embedding_model": EMBEDDING_MODEL,
            "top_matches": detailed_results,
            "all_similarity_scores": [
                {
                    "resume_id": match.resume_id,
                    "jd_id": match.jd_id,
                    "similarity": match.similarity_score
                } for match in all_matches
            ]
        }
        
        print(f"\n[SUCCESS] Comparison completed in {final_results['processing_time_seconds']} seconds")
        return final_results

    def run_targeted_search(self, query: str, search_type: str = "both", k: int = 5) -> Dict[str, Any]:
        """Run targeted search for specific skills, roles, or requirements"""
        print(f"\n[INFO] Running targeted search for: '{query}'")

        results = {
            "query": query,
            "search_type": search_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": []
        }

        if search_type in ["resume", "both"]:
            print("[INFO] Searching resumes...")
            resume_matches = self.resume_store.similarity_search_with_score(query, k=k)
            for doc, score in resume_matches:
                results["results"].append({
                    "type": "resume",
                    "id": doc.metadata.get('source', 'unknown'),
                    "similarity_score": float(score),
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })

        if search_type in ["jd", "both"]:
            print("[INFO] Searching job descriptions...")
            jd_matches = self.jd_store.similarity_search_with_score(query, k=k)
            for doc, score in jd_matches:
                results["results"].append({
                    "type": "job_description",
                    "id": doc.metadata.get('source', 'unknown'),
                    "similarity_score": float(score),
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })

        # Sort all results by similarity score
        results["results"].sort(key=lambda x: x["similarity_score"], reverse=True)

        return results

def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description="Embedding-based Resume-JD Comparison")
    parser.add_argument("--mode", choices=["full", "search"], default="full",
                       help="Run full comparison or targeted search")
    parser.add_argument("--query", type=str, help="Search query for targeted search mode")
    parser.add_argument("--search-type", choices=["resume", "jd", "both"], default="both",
                       help="Type of documents to search")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD,
                       help="Similarity threshold for matches")
    parser.add_argument("--top-k", type=int, default=10,
                       help="Number of top matches to analyze in detail")

    args = parser.parse_args()

    print("Starting Embedding-Based Resume-JD Comparison System")
    print("-" * 50)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize comparator
    comparator = EmbeddingComparator(similarity_threshold=args.threshold)

    if args.mode == "search":
        if not args.query:
            print("[ERROR] Query is required for search mode!")
            return

        print(f"[INFO] Running targeted search mode")
        results = comparator.run_targeted_search(args.query, args.search_type)
        output_file = os.path.join(OUTPUT_DIR, f"search_results_{int(time.time())}.json")

        # Print search results
        print(f"\nSEARCH RESULTS for '{args.query}':")
        print("-" * 40)
        for i, result in enumerate(results["results"][:10]):
            print(f"{i+1}. [{result['type'].upper()}] {result['id']}")
            print(f"   Similarity: {result['similarity_score']:.4f}")
            print(f"   Content preview: {result['content'][:100]}...")
            print()

    else:
        print(f"[INFO] Running full comparison mode")
        results = comparator.run_comprehensive_comparison()
        output_file = os.path.join(OUTPUT_DIR, "embedding_comparison_results.json")

        # Print summary
        if "error" not in results and "warning" not in results:
            print(f"\nSUMMARY:")
            print(f"- Total resumes analyzed: {results['total_resumes']}")
            print(f"- Total job descriptions: {results['total_jds']}")
            print(f"- Matches above threshold: {results['total_matches_above_threshold']}")
            print(f"- Processing time: {results['processing_time_seconds']} seconds")

            if results['top_matches']:
                print(f"\nALL MATCHES (sorted by similarity):")
                for i, match in enumerate(results['top_matches']):
                    print(f"{i+1}. {match['resume_id']} vs {match['jd_id']}")
                    print(f"   Embedding Similarity: {match['embedding_similarity']:.4f}")

                    # Show overall match percentage
                    if 'OverallMatchPercentage' in match['detailed_analysis']:
                        overall_pct = match['detailed_analysis']['OverallMatchPercentage']
                        print(f"   Overall Match: {overall_pct:.1%}")
                    elif 'overall_match' in match['detailed_analysis']:
                        overall_score = match['detailed_analysis']['overall_match']['score']
                        print(f"   Overall Score: {overall_score:.1f}%")

                    # Show individual category scores
                    analysis = match['detailed_analysis']
                    if 'Skills' in analysis:
                        print(f"   Skills: {analysis['Skills']['match_pct']:.1%}")
                    if 'Experience' in analysis:
                        print(f"   Experience: {analysis['Experience']['match_pct']:.1%}")
                    if 'Education' in analysis:
                        print(f"   Education: {analysis['Education']['match_pct']:.1%}")
                    if 'Job Role' in analysis:
                        print(f"   Job Role: {analysis['Job Role']['match_pct']:.1%}")

                    print()

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Results saved to: {output_file}")

if __name__ == "__main__":
    main()
