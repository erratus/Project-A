import os
import json
import time
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# === Paths and Config ===
resume_json_dir = "resume_json"
jd_json_dir = "JD_extraction"
model_name = "bigllama/mistralv01-7b:latest"

system_prompt = """You are a world-class HR, Talent Acquisition, and Generative AI Specialist with deep expertise in job-role alignment, semantic document comparison, and hiring decision automation.
You are tasked with comparing a candidate resume and a job description. Both are pre-parsed into structured fields: Skills, Education, Job Role, Experience, and Other Information. Your job is to assess the alignment strictly based on meaning — not exact keyword matches.
You must return a single valid JSON object in the structure described below.
 
Instructions:
- Evaluate semantic relevance, not keyword overlap. For example, treat "ML Engineer" and "Machine Learning Engineer" as equivalent.
- Apply real-world hiring logic: If the resume exceeds JD requirements (e.g., more skills, higher education, deeper experience), assign a high match_pct — even 100%.
- Do not penalize minor differences in naming or formatting.
- Never assign 0% if a field contains any valid data. Only assign 0% if the resume field is empty or clearly unrelated.
- Never hallucinate or infer information not present in either document.
- Never nest objects — keep JSON flat.
- Escape invalid characters like \\t, \\n, and quotes properly.
- Use consistent, professional phrasing in all explanations.
 
Field Matching Logic:
Skills
- Match based on technical equivalence, not wording.
- If the resume includes all required skills (or more), assign 100%.
- If semantically similar (e.g., "pandas" vs. "Python data manipulation"), assign 80–95%.
- Penalize only if critical skills are missing.
- If no overlap at all, assign <30% with explanation.
 
Education
- Full match (100%) if the degree level and field align with the JD.
- Slight name variations are acceptable (B.E. ≈ B.Tech).
- If field is unrelated (e.g., BA in History for Data Scientist JD), assign 20–30%.
- If degree level is below requirement (e.g., diploma instead of B.Tech), assign <30%.
- Clearly explain penalties.
 
Experience
- Match on role relevance, years of experience, technologies used, domain familiarity.
- Resume that meets or exceeds JD’s experience should score 90–100%.
- Penalize only if domain is different, role is mismatched, or years are far below JD.
 
Job Role
- Normalize semantically similar roles: "ML Engineer" ≈ "Machine Learning Engineer"
- If role meaning aligns, score 90–100%.
- Assign <50% only if roles are clearly different (e.g., “Product Manager” vs. “ML Engineer”).
- Only assign 0% if the resume job role field is empty.
 
OverallMatchPercentage
- Weighted average of: Skills (30%), Experience (30%), Education (20%), Job Role (20%)
- Add/subtract ±5% for “Other Information” if highly relevant or problematic.
- Clearly explain rationale for final score.
 
AI_Generated_Estimate_Percentage
- High score (80–100%) if language is overly perfect, repetitive, generic.
- Low (0–30%) if nuanced, varied, clearly human-written.
 
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
 
- Output only the JSON, no markdown, no extra commentary.
- Do not change key order or structure.."""

user_prompt_template = """You are tasked with comparing a candidate’s resume and a job description, each parsed into structured fields: Skills, Education, Job Role, Experience, and Other Information.
Your goal is to evaluate the semantic alignment between the resume and the job description — focusing on meaning and capability, not exact keyword matches.
Return only a single valid JSON object in this exact structure (replace {resume_filename} with the actual resume file name):

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

Evaluation Instructions:
Semantic similarity means matching based on meaning, not exact text. For example, treat "ML Engineer" and "Machine Learning Engineer" as equivalent that mean both are same .
Use realistic hiring logic: if the resume exceeds JD requirements (more skills, higher education, deeper experience), assign high or full match_pct.
Assign match_pct scores from 0 to 100 representing the degree of semantic alignment.
The overall percentage calculated based on Skill,Education,Experience,Job Role majorly and minorly from other Information
Avoid penalizing minor wording or formatting differences.
Penalize missing key required fields appropriately.
No semicolons (;) in values — use periods or commas
For missing or partial data in any field, explain clearly how it impacts match_pct.
Do not hallucinate or infer data not explicitly present in either document.
Never nest JSON objects inside any field values; keep all fields flat.
Escape special characters (tabs, newlines, quotes) using \\t, \\n, \" respectively.
Maintain consistent phrasing and tone in explanations, as if writing to a hiring manager.
Do not output any text or commentary outside the JSON object.
Output:
Return only the JSON object following the structure above. Do not add any extra text or commentary."""

def load_json_files(directory, file_type):
    """Load all JSON files from a directory"""
    files = []
    if not os.path.exists(directory):
        print(f"[ERROR] Directory {directory} does not exist!")
        return files
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['filename'] = filename
                    files.append(data)
                    print(f"[INFO] Loaded {file_type}: {filename}")
            except Exception as e:
                print(f"[ERROR] Failed to load {filename}: {e}")
    
    return files

def format_field_data(data, field_name):
    """Format field data for display"""
    field_data = data.get(field_name, [])
    if isinstance(field_data, list):
        return ', '.join(field_data) if field_data else "Not specified"
    return str(field_data) if field_data else "Not specified"

# === Load JSON Files ===
print("[INFO] Loading JSON files...")
resume_files = load_json_files(resume_json_dir, "resume")
jd_files = load_json_files(jd_json_dir, "job description")

# Validate that we have data
if not jd_files:
    print("[ERROR] No job descriptions found!")
    exit(1)

if not resume_files:
    print("[ERROR] No resumes found!")
    exit(1)

jd = jd_files[0]  # assume only 1 JD
print(f"[INFO] Using JD: {jd.get('filename', 'unknown')}")
print(f"[INFO] Found {len(resume_files)} resumes to compare")

chat = ChatOllama(model=model_name, temperature=0.0, seed=42)

def run_comparison(resume, jd, resume_filename):
    """Run comparison between resume and job description"""
    
    # Format the data for comparison
    resume_skills = format_field_data(resume, 'skill')
    resume_education = format_field_data(resume, 'education')
    resume_experience = format_field_data(resume, 'experience')
    resume_job_role = format_field_data(resume, 'job role')
    resume_other = format_field_data(resume, 'other information')
    
    jd_skills = format_field_data(jd, 'skill')
    jd_education = format_field_data(jd, 'education')
    jd_experience = format_field_data(jd, 'experience')
    jd_job_role = format_field_data(jd, 'job role')
    jd_other = format_field_data(jd, 'other information')
    
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
- Other Information: {jd_other}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    try:
        print(f"[INFO] Processing resume: {resume_filename}")
        response = chat.invoke(messages)
        content = response.content.strip()
        
        # Clean up the response
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
                print(f"[DEBUG] JSON content end: {json_content[-300:]}")

                # First, try to find if we have the overall fields in the original content
                original_content = content
                overall_match = None
                why_overall = None
                ai_estimate = None

                # Extract overall fields from the original content if they exist
                import re
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

                        print(f"[DEBUG] Repaired JSON keys: {list(result.keys())}")
                        if resume_filename in result:
                            print(f"[DEBUG] Result keys for {resume_filename}: {list(result[resume_filename].keys())}")
                    except json.JSONDecodeError as repair_error:
                        # If still failing, try a more aggressive approach
                        print(f"[WARNING] JSON repair failed: {repair_error}")
                        print(f"[DEBUG] Attempted repair content: {json_content[-200:]}")
                        raise e
                else:
                    raise e

            print(f"[SUCCESS] Successfully processed {resume_filename}")
            print(f"[DEBUG] Final result keys: {list(result.keys())}")
            if resume_filename in result:
                print(f"[DEBUG] Data keys for {resume_filename}: {list(result[resume_filename].keys())}")
            return result
        else:
            print(f"[ERROR] No valid JSON found in response for {resume_filename}")
            return {resume_filename: {"error": "No valid JSON found in LLM response", "raw_response": content}}
            
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing failed for {resume_filename}: {e}")
        print(f"[DEBUG] Raw response length: {len(content)}")
        print(f"[DEBUG] First 500 chars: {content[:500]}")
        print(f"[DEBUG] Last 500 chars: {content[-500:]}")

        # Try to save the problematic response for debugging
        debug_file = f"debug_response_{resume_filename.replace('.json', '')}.txt"
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[DEBUG] Full response saved to: {debug_file}")

        return {resume_filename: {"error": f"JSON parsing failed: {str(e)}", "raw_response": content[:1000]}}
    except Exception as e:
        print(f"[ERROR] General error for {resume_filename}: {e}")
        return {resume_filename: {"error": str(e)}}

# === Run Comparisons ===
print(f"\n[INFO] Starting comparison process...")
print(f"[INFO] Job Description: {jd.get('filename', 'unknown')}")
print(f"[INFO] Number of resumes to process: {len(resume_files)}")
print("-" * 50)

results = {}
successful_matches = 0
failed_matches = 0

for i, resume in enumerate(resume_files, 1):
    fname = resume.get("filename", f"resume_{int(time.time())}.json")
    print(f"\n[{i}/{len(resume_files)}] Processing: {fname}")
    
    match = run_comparison(resume, jd, fname)
    results.update(match)
    
    # Check if the match was successful
    if fname in match and "error" not in match[fname]:
        successful_matches += 1
        print(f"[SUCCESS] Completed: {fname}")
    else:
        failed_matches += 1
        print(f"[FAILED] Error processing: {fname}")

print("\n" + "=" * 50)
print(f"[SUMMARY] Processing complete!")
print(f"[SUMMARY] Successful matches: {successful_matches}")
print(f"[SUMMARY] Failed matches: {failed_matches}")
print(f"[SUMMARY] Total processed: {len(resume_files)}")

# === Save Results in format.json structure ===
os.makedirs("output", exist_ok=True)
output_file = "output/resume_jd_matches_direct_single_pass2.json"

# Convert results to format.json structure
formatted_results = []
jd_name = jd.get("filename", "JobDescription").replace(".json", "")

for resume_name, data in results.items():
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
                "AI_Generated_Estimate_Percentage": data.get("AI_Generated_Estimate_Percentage", 0.0)
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
                "AI_Generated_Estimate_Percentage": 0.0
            }
        }
        formatted_results.append(error_entry)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(formatted_results, f, indent=2, ensure_ascii=False)

print(f"\n[INFO] Results saved to: {output_file}")
print(f"[INFO] Format matches: format.json structure")

# Print detailed results in terminal
print(f"\n[INFO] DETAILED MATCH RESULTS:")
print("=" * 80)

for resume_name, data in results.items():
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
        print(f"\n🎯 OVERALL MATCH: {format_percentage(overall_pct)}")
        print(f"   Reasoning: {data.get('why_overall_match_is_this', 'N/A')}")
        print(f"   AI Generated Estimate: {format_percentage(data.get('AI_Generated_Estimate_Percentage', 0))}")

    else:
        print(f"\n❌ ERROR for {resume_name}: {data.get('error', 'Unknown error')}")

print(f"\n" + "=" * 80)
print(f"Direct JSON comparison completed successfully!")
