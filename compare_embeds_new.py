#!/usr/bin/env python3
"""
Embedding-based Resume-JD Comparison Script

This script performs comparison between resumes and job descriptions using embeddings
but maintains the same behavior and output format as compare_direct.py.
"""

import os
import json
import time
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from sklearn.metrics.pairwise import cosine_similarity

# === Configuration ===
RESUME_EMBEDDINGS_BASE_DIR = "chroma_db_resumes"
JD_EMBEDDINGS_DIR = "chroma_db_jd"
JD_EXTRACTION_DIR = "JD_extraction"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "llama3.2:latest"

# === System Prompt (updated for accurate scoring) ===
system_prompt = """You are a world-class HR, Talent Acquisition, and Generative AI Specialist with deep expertise in job-role alignment, semantic document comparison, and hiring decision automation.

You are tasked with comparing a candidate resume and a job description. Both are pre-parsed into structured fields: Skills, Education, Job Role, Experience, and Other Information. Your job is to assess the alignment based on realistic hiring standards and practical job fit — not just keyword overlap.

You must return a single valid JSON object in the structure described below.

SCORING GUIDELINES - Apply realistic hiring standards:

Instructions:
- Evaluate semantic relevance and domain expertise, not just keyword overlap.
- Apply real-world hiring logic: Score based on actual fit for the role.
- Never assign 0% if a field contains any valid data.
- Never hallucinate or infer information not present in either document.
- Never nest objects — keep JSON flat.
- Use consistent, professional phrasing in all explanations.
- Be honest about mismatches - don't inflate scores for unrelated fields.

CRITICAL EDUCATION EXAMPLES:
- "B.Tech in Computer Science" vs "B Tech/M Tech in Computer/IT" = 95% match (exact field, minor name variation)
- "B.E. in Information Technology" vs "B Tech in Computer Science" = 95% match (same field, different degree names)
- "M.Tech in Computer Science" vs "B Tech/M Tech in Computer/IT" = 100% match (exact match)

Field Matching Logic:

Skills Scoring:
- 90-100%: Candidate has most/all required skills with strong domain expertise
- 80-89%: Candidate has many required skills with good domain knowledge
- 70-79%: Candidate has some required skills with moderate relevance
- 50-69%: Candidate has few required skills but some transferable knowledge
- 30-49%: Candidate has minimal relevant skills
- 10-29%: Candidate has very few or tangentially related skills
- 0-9%: No relevant skills found

Education Scoring:
- 90-100%: Degree field exactly matches JD requirements (e.g., B.Tech/B.E./M.Tech in CS/IT/Computer Science for tech roles). Minor variations in degree names are acceptable (B.Tech ≈ B Tech, Computer Science ≈ Computer/IT ≈ IT). Institution reputation is secondary to field match.
- 80-89%: Degree field closely related to JD requirements (e.g., Electronics/ECE for tech roles) from any institution
- 70-79%: Degree field somewhat related (e.g., any Engineering for tech roles) or exact match from lesser-known institution
- 50-69%: Degree field moderately related (e.g., Science/Mathematics for tech roles) from any institution
- 30-49%: Degree field distantly related (e.g., Business for tech roles) but from excellent institution
- 10-29%: Degree field unrelated (e.g., Arts/Humanities for tech roles) from any institution
- 0-9%: No degree information or completely irrelevant education

IMPORTANT: For Education matching, prioritize FIELD ALIGNMENT over institution name. B.Tech in Computer Science should score 90-100% for "B Tech/M Tech in Computer/IT" requirements regardless of institution.

Experience Scoring:
- 90-100%: Years match JD requirements with highly relevant domain experience
- 80-89%: Years match with good domain relevance
- 70-79%: Years slightly below/above with good domain relevance, or years match with moderate relevance
- 50-69%: Years significantly different but domain is relevant, or years match but domain is moderately relevant
- 30-49%: Years and domain both have moderate gaps
- 10-29%: Significant gaps in both years and domain relevance
- 0-9%: Major misalignment in experience requirements

Job Role Scoring:
- 90-100%: Current/recent role is the same or very similar to target role
- 80-89%: Current role is closely related with significant overlap
- 70-79%: Current role has good overlap with target role
- 50-69%: Current role has moderate relevance to target role
- 30-49%: Current role has some transferable skills to target role
- 10-29%: Current role has minimal relevance to target role
- 0-9%: Current role is completely unrelated to target role

OverallMatchPercentage Calculation:
- Weighted average of: Skills (30%), Experience (30%), Education (20%), Job Role (20%)
- Add/subtract ±5% for "Other Information" if highly relevant or problematic.
- Clearly explain rationale for final score.
- Be realistic - don't inflate scores for poor matches.

AI_Generated_Estimate_Percentage:
- High score (80–100%) if language is overly perfect, repetitive, uses generic buzzwords, lacks specific details, has unnatural phrasing, or contains obvious AI patterns like "leveraging", "cutting-edge", "state-of-the-art" repeatedly.
- Medium score (40–79%) if language is polished but contains specific technical details, project outcomes, or personal experiences that suggest human authorship.
- Low score (0–39%) if language is nuanced, varied, contains specific metrics/achievements, has natural imperfections, or shows clear human personality/voice.

IMPORTANT: Technical resumes with specific project details, metrics (like "MSE = 0.032"), company names, and concrete achievements should score LOW on AI detection, even if well-written.

Output Format (Strict JSON Only):
{
  "{resume_filename}": {
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
}

Return ONLY the JSON object. No extra comments or explanation."""

user_prompt_template = """Compare the following resume and job description using their parsed field data.

Each field below is populated from the database. Compare them **semantically and intelligently** using the structure below.

Use the following strict JSON format:

{{
  "{resume_filename}": {{
    "Skills": {{
      "match_pct": float,
      "resume_value": string,
      "job_description_value": string,
      "explanation": string
    }},
    "Education": {{
      "match_pct": float,
      "resume_value": string,
      "job_description_value": string,
      "explanation": string
    }},
    "Job Role": {{
      "match_pct": float,
      "resume_value": string,
      "job_description_value": string,
      "explanation": string
    }},
    "Experience": {{
      "match_pct": float,
      "resume_value": string,
      "job_description_value": string,
      "explanation": string
    }},
    "OverallMatchPercentage": float,
    "why_overall_match_is_this": string,
    "AI_Generated_Estimate_Percentage": float
  }}
}}

Return only the JSON object, and ensure:
- match_pct values reflect real semantic similarity (not keyword count).
- Explanations are professional, specific, and insightful.
- No nested JSON objects inside any value fields.
- No semicolons (;) in values — use periods or commas.
- No hallucinated info or missing keys."""

class EmbeddingBasedComparator:
    """Main class for embedding-based resume-JD comparison"""
    
    def __init__(self):
        """Initialize the comparator"""
        print("[INFO] Initializing Embedding-Based Comparator...")
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize LLM for detailed analysis
        self.chat = ChatOllama(model=MODEL_NAME, temperature=0.0, seed=42)
        
        print("[INFO] Initialization complete!")
    
    def load_resume_data(self, resume_folder_path: str) -> Dict[str, Any]:
        """Load resume data from its embedding folder"""
        # Load original JSON data
        json_file_path = os.path.join(resume_folder_path, "original_data.json")
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"Original data not found: {json_file_path}")
        
        with open(json_file_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)
        
        # Load embeddings
        try:
            vectorstore = Chroma(
                persist_directory=resume_folder_path,
                embedding_function=self.embedding_model
            )
            # Get the document
            docs = vectorstore.similarity_search(".", k=1)
            if docs:
                embedding_content = docs[0].page_content
                metadata = docs[0].metadata
            else:
                embedding_content = ""
                metadata = {}
        except Exception as e:
            print(f"[WARNING] Could not load embeddings from {resume_folder_path}: {e}")
            embedding_content = ""
            metadata = {}
        
        return {
            "original_data": original_data,
            "embedding_content": embedding_content,
            "metadata": metadata,
            "filename": metadata.get("source", "unknown")
        }
    
    def load_jd_data(self) -> Dict[str, Any]:
        """Load job description data from embeddings"""
        # Load from JD extraction directory
        jd_files = [f for f in os.listdir(JD_EXTRACTION_DIR) if f.endswith(".json")]
        if not jd_files:
            raise FileNotFoundError("No JD files found in extraction directory")
        
        # Use the first JD file (assuming single JD for now)
        jd_file = jd_files[0]
        jd_path = os.path.join(JD_EXTRACTION_DIR, jd_file)
        
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_data = json.load(f)
        
        # Load JD embeddings
        try:
            vectorstore = Chroma(
                persist_directory=JD_EMBEDDINGS_DIR,
                embedding_function=self.embedding_model
            )
            docs = vectorstore.similarity_search(".", k=1)
            if docs:
                embedding_content = docs[0].page_content
                metadata = docs[0].metadata
            else:
                embedding_content = ""
                metadata = {}
        except Exception as e:
            print(f"[WARNING] Could not load JD embeddings: {e}")
            embedding_content = ""
            metadata = {}
        
        return {
            "original_data": jd_data,
            "embedding_content": embedding_content,
            "metadata": metadata,
            "filename": jd_file
        }
    
    def calculate_embedding_similarity(self, resume_content: str, jd_content: str) -> float:
        """Calculate cosine similarity between resume and JD embeddings"""
        try:
            resume_embedding = self.embedding_model.embed_query(resume_content)
            jd_embedding = self.embedding_model.embed_query(jd_content)
            
            similarity = cosine_similarity([resume_embedding], [jd_embedding])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"[WARNING] Could not calculate embedding similarity: {e}")
            return 0.0
    
    def format_field_data(self, data: Dict, field_name: str) -> str:
        """Format field data for comparison (same as compare_direct.py)"""
        field_data = data.get(field_name, [])
        if isinstance(field_data, list):
            return ", ".join(field_data) if field_data else f"No {field_name} information provided"
        return str(field_data) if field_data else f"No {field_name} information provided"

    def run_comparison(self, resume_data: Dict, jd_data: Dict, resume_filename: str) -> Dict[str, Any]:
        """Run comparison between resume and JD using embeddings + LLM analysis"""

        # Calculate embedding similarity for context
        embedding_similarity = self.calculate_embedding_similarity(
            resume_data["embedding_content"],
            jd_data["embedding_content"]
        )

        print(f"[INFO] Embedding similarity: {embedding_similarity:.4f}")

        # Format the data for LLM comparison (same as compare_direct.py)
        resume_skills = self.format_field_data(resume_data["original_data"], 'skill')
        resume_education = self.format_field_data(resume_data["original_data"], 'education')
        resume_experience = self.format_field_data(resume_data["original_data"], 'experience')
        resume_job_role = self.format_field_data(resume_data["original_data"], 'job role')
        resume_other = self.format_field_data(resume_data["original_data"], 'other information')

        jd_skills = self.format_field_data(jd_data["original_data"], 'skill')
        jd_education = self.format_field_data(jd_data["original_data"], 'education')
        jd_experience = self.format_field_data(jd_data["original_data"], 'experience')
        jd_job_role = self.format_field_data(jd_data["original_data"], 'job role')
        jd_other = self.format_field_data(jd_data["original_data"], 'other information')

        user_prompt = user_prompt_template.format(
            resume_filename=resume_filename
        ) + f"""

Resume Data:
- Skills: {resume_skills}
- Education: {resume_education}
- Experience: {resume_experience}
- Job Role: {resume_job_role}
- Other Information: {resume_other}

Job Description Data:
- Skills: {jd_skills}
- Education: {jd_education}
- Experience: {jd_experience}
- Job Role: {jd_job_role}
- Other Information: {jd_other}

Additional Context:
- Embedding Similarity Score: {embedding_similarity:.4f} (indicates semantic alignment)"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        try:
            print(f"[INFO] Processing resume: {resume_filename}")
            response = self.chat.invoke(messages)
            content = response.content.strip()

            # Clean up the response (same logic as compare_direct.py)
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            elif content.startswith('```'):
                content = content.replace('```', '').strip()

            # Find JSON object
            json_start = content.find('{')
            json_end = content.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_content = content[json_start:json_end]

                # Fix common JSON escape issues
                json_content = json_content.replace('\\_', '_')
                json_content = json_content.replace('\\n', '\\\\n')
                json_content = json_content.replace('\\t', '\\\\t')

                # Try to fix incomplete JSON by adding missing closing braces
                try:
                    result = json.loads(json_content)
                except json.JSONDecodeError as e:
                    print(f"[WARNING] Initial JSON parse failed, attempting repair...")
                    print(f"[DEBUG] Original error: {e}")

                    # Extract overall fields from the original content if they exist
                    original_content = content
                    overall_match = None
                    why_overall = None
                    ai_estimate = None

                    # Extract overall fields from the original content if they exist
                    overall_pattern = r'"OverallMatchPercentage":\s*(\d+(?:\.\d+)?)'
                    why_pattern = r'"why_overall_match_is_this":\s*"([^"]*)"'
                    ai_pattern = r'"AI_Generated_Estimate_Percentage":\s*(\d+(?:\.\d+)?)'

                    overall_match_obj = re.search(overall_pattern, original_content)
                    why_match_obj = re.search(why_pattern, original_content)
                    ai_match_obj = re.search(ai_pattern, original_content)

                    if overall_match_obj:
                        overall_match = float(overall_match_obj.group(1))
                        print(f"[DEBUG] Found OverallMatchPercentage: {overall_match}")

                    if why_match_obj:
                        why_overall = why_match_obj.group(1)
                        print(f"[DEBUG] Found why_overall_match_is_this: {why_overall[:50]}...")

                    if ai_match_obj:
                        ai_estimate = float(ai_match_obj.group(1))
                        print(f"[DEBUG] Found AI_Generated_Estimate_Percentage: {ai_estimate}")

                    # Count opening and closing braces
                    open_braces = json_content.count('{')
                    close_braces = json_content.count('}')

                    if open_braces > close_braces:
                        # Add missing closing braces
                        missing_braces = open_braces - close_braces
                        json_content += '}' * missing_braces
                        print(f"[INFO] Added {missing_braces} missing closing braces")

                        try:
                            result = json.loads(json_content)
                            print(f"[SUCCESS] JSON repair successful for {resume_filename}")

                            # Add the overall fields if they were found but missing from parsed result
                            if resume_filename in result:
                                if overall_match is not None and 'OverallMatchPercentage' not in result[resume_filename]:
                                    result[resume_filename]['OverallMatchPercentage'] = overall_match
                                    print(f"[INFO] Added missing OverallMatchPercentage: {overall_match}")

                                if why_overall is not None and 'why_overall_match_is_this' not in result[resume_filename]:
                                    result[resume_filename]['why_overall_match_is_this'] = why_overall
                                    print(f"[INFO] Added missing why_overall_match_is_this")

                                if ai_estimate is not None and 'AI_Generated_Estimate_Percentage' not in result[resume_filename]:
                                    result[resume_filename]['AI_Generated_Estimate_Percentage'] = ai_estimate
                                    print(f"[INFO] Added missing AI_Generated_Estimate_Percentage: {ai_estimate}")

                                # Add embedding similarity to the result
                                result[resume_filename]['embedding_similarity'] = embedding_similarity

                        except json.JSONDecodeError as repair_error:
                            print(f"[WARNING] JSON repair failed: {repair_error}")
                            raise e
                    else:
                        raise e

                # Add embedding similarity to successful results
                if resume_filename in result:
                    result[resume_filename]['embedding_similarity'] = embedding_similarity

                print(f"[SUCCESS] Successfully processed {resume_filename}")
                return result
            else:
                print(f"[ERROR] No valid JSON found in response for {resume_filename}")
                return {resume_filename: {"error": "No valid JSON found in LLM response", "raw_response": content}}

        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON parsing failed for {resume_filename}: {e}")
            return {resume_filename: {"error": f"JSON parsing failed: {str(e)}", "raw_response": content[:1000]}}
        except Exception as e:
            print(f"[ERROR] General error for {resume_filename}: {e}")
            return {resume_filename: {"error": str(e)}}

    def find_resume_folders(self) -> List[str]:
        """Find all resume embedding folders"""
        if not os.path.exists(RESUME_EMBEDDINGS_BASE_DIR):
            print(f"[ERROR] Resume embeddings directory not found: {RESUME_EMBEDDINGS_BASE_DIR}")
            return []

        folders = []
        for item in os.listdir(RESUME_EMBEDDINGS_BASE_DIR):
            item_path = os.path.join(RESUME_EMBEDDINGS_BASE_DIR, item)
            if os.path.isdir(item_path):
                folders.append(item_path)

        print(f"[INFO] Found {len(folders)} resume embedding folders")
        return folders

def main():
    """Main execution function"""
    print("[INFO] Starting Embedding-Based Resume-JD Comparison...")
    print("[INFO] This script uses embeddings but maintains the same output format as compare_direct.py")

    # Initialize comparator
    comparator = EmbeddingBasedComparator()

    # Load JD data
    try:
        jd_data = comparator.load_jd_data()
        print(f"[INFO] Loaded JD: {jd_data['filename']}")
    except Exception as e:
        print(f"[ERROR] Failed to load JD data: {e}")
        return

    # Find resume folders
    resume_folders = comparator.find_resume_folders()
    if not resume_folders:
        print("[ERROR] No resume embedding folders found!")
        return

    print(f"[INFO] Found {len(resume_folders)} resumes to compare")

    # Process each resume
    all_results = {}
    successful_matches = 0
    failed_matches = 0

    print("\n" + "="*50)
    print("[INFO] Starting comparison process...")
    print(f"[INFO] Job Description: {jd_data['filename']}")
    print(f"[INFO] Number of resumes to process: {len(resume_folders)}")
    print("-" * 50)

    for i, resume_folder in enumerate(resume_folders):
        try:
            # Load resume data
            resume_data = comparator.load_resume_data(resume_folder)
            resume_filename = resume_data['filename']

            print(f"\n[{i+1}/{len(resume_folders)}] Processing: {resume_filename}")

            # Run comparison
            result = comparator.run_comparison(resume_data, jd_data, resume_filename)

            # Merge results
            all_results.update(result)

            if "error" not in result.get(resume_filename, {}):
                successful_matches += 1
                print(f"[SUCCESS] Completed: {resume_filename}")
            else:
                failed_matches += 1
                print(f"[FAILED] Error processing: {resume_filename}")

        except Exception as e:
            failed_matches += 1
            print(f"[FAILED] Exception processing {resume_folder}: {e}")
            folder_name = os.path.basename(resume_folder)
            all_results[f"{folder_name}.json"] = {"error": str(e)}

    # === Save Results in format.json structure ===
    os.makedirs("single_output", exist_ok=True)
    output_file = "single_output/single_embed_pass_2.json"

    # Convert results to format.json structure
    formatted_results = []
    jd_name = jd_data["filename"].replace(".json", "")

    for resume_name, data in all_results.items():
        if "error" not in data:
            candidate_name = resume_name.replace(".json", "")
            key = f"{candidate_name}_vs_{jd_name}"

            formatted_entry = {
                key: {
                    "Skills": data.get("Skills", {}),
                    "Education": data.get("Education", {}),
                    "Job Role": data.get("Job Role", {}),
                    "Experience": data.get("Experience", {}),
                    "OverallMatchPercentage": data.get("OverallMatchPercentage", 0.0),
                    "why_overall_match_is_this": data.get("why_overall_match_is_this", ""),
                    "AI_Generated_Estimate_Percentage": data.get("AI_Generated_Estimate_Percentage", 0.0),
                    "embedding_similarity": data.get("embedding_similarity", 0.0)
                }
            }
            formatted_results.append(formatted_entry)
        else:
            # Include errors in a separate format
            candidate_name = resume_name.replace(".json", "")
            key = f"{candidate_name}_vs_{jd_name}"
            error_entry = {
                key: {
                    "error": data.get("error", "Unknown error"),
                    "Skills": {"match_pct": 0.0, "resume_value": "", "job_description_value": "", "explanation": "Error occurred"},
                    "Education": {"match_pct": 0.0, "resume_value": "", "job_description_value": "", "explanation": "Error occurred"},
                    "Job Role": {"match_pct": 0.0, "resume_value": "", "job_description_value": "", "explanation": "Error occurred"},
                    "Experience": {"match_pct": 0.0, "resume_value": "", "job_description_value": "", "explanation": "Error occurred"},
                    "OverallMatchPercentage": 0.0,
                    "why_overall_match_is_this": f"Analysis failed due to error: {data.get('error', 'Unknown error')}",
                    "AI_Generated_Estimate_Percentage": 0.0,
                    "embedding_similarity": 0.0
                }
            }
            formatted_results.append(error_entry)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_results, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Results saved to: {output_file}")
    print(f"[INFO] Format matches: format.json structure")

    # Print detailed results in terminal (same as compare_direct.py)
    print(f"\n[INFO] DETAILED MATCH RESULTS:")
    print("=" * 80)

    for resume_name, data in all_results.items():
        if "error" not in data:
            print(f"\n📋 CANDIDATE: {resume_name}")
            print("-" * 60)

            # Helper function to format percentage (handles both 0-1 and 0-100 ranges)
            def format_percentage(value):
                if isinstance(value, (int, float)):
                    if value > 1:  # Assume it's already in 0-100 range
                        return f"{value:.1f}%"
                    else:  # Assume it's in 0-1 range
                        return f"{value:.1%}"
                return "0.0%"

            # Skills
            if "Skills" in data:
                skills = data["Skills"]
                print(f"🔧 SKILLS MATCH: {format_percentage(skills.get('match_pct', 0))}")
                print(f"   Resume Skills: {skills.get('resume_value', 'N/A')[:100]}...")
                print(f"   Required Skills: {skills.get('job_description_value', 'N/A')}")
                print(f"   Explanation: {skills.get('explanation', 'N/A')}")

            # Education
            if "Education" in data:
                education = data["Education"]
                print(f"\n🎓 EDUCATION MATCH: {format_percentage(education.get('match_pct', 0))}")
                print(f"   Resume Education: {education.get('resume_value', 'N/A')}")
                print(f"   Required Education: {education.get('job_description_value', 'N/A')}")
                print(f"   Explanation: {education.get('explanation', 'N/A')}")

            # Job Role
            if "Job Role" in data:
                role = data["Job Role"]
                print(f"\n💼 JOB ROLE MATCH: {format_percentage(role.get('match_pct', 0))}")
                print(f"   Resume Role: {role.get('resume_value', 'N/A')}")
                print(f"   Target Role: {role.get('job_description_value', 'N/A')}")
                print(f"   Explanation: {role.get('explanation', 'N/A')}")

            # Experience
            if "Experience" in data:
                experience = data["Experience"]
                print(f"\n💻 EXPERIENCE MATCH: {format_percentage(experience.get('match_pct', 0))}")
                print(f"   Resume Experience: {experience.get('resume_value', 'N/A')[:150]}...")
                print(f"   Required Experience: {experience.get('job_description_value', 'N/A')[:150]}...")
                print(f"   Explanation: {experience.get('explanation', 'N/A')}")

            # Overall Results
            overall_pct = data.get("OverallMatchPercentage", 0)
            embedding_sim = data.get("embedding_similarity", 0)
            print(f"\n🎯 OVERALL MATCH: {format_percentage(overall_pct)}")
            print(f"   Reasoning: {data.get('why_overall_match_is_this', 'N/A')}")
            print(f"   AI Generated Estimate: {format_percentage(data.get('AI_Generated_Estimate_Percentage', 0))}")
            print(f"   🔗 Embedding Similarity: {embedding_sim:.4f}")

        else:
            print(f"\n❌ ERROR for {resume_name}: {data.get('error', 'Unknown error')}")

    print(f"\n" + "="*50)
    print(f"[SUMMARY] Processing complete!")
    print(f"[SUMMARY] Successful matches: {successful_matches}")
    print(f"[SUMMARY] Failed matches: {failed_matches}")
    print(f"[SUMMARY] Total processed: {len(resume_folders)}")
    print(f"\nEmbedding-based comparison completed successfully!")

if __name__ == "__main__":
    main()
