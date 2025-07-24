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

You are tasked with comparing a candidate resume and a job description. Both are pre-parsed into structured fields: Skills, Education, Job Role, Experience, and Other Information. Your job is to assess the alignment based on **realistic hiring standards** and **practical job fit** — not just keyword overlap.

You must return a single valid JSON object in the structure described below.

### Instructions:

- Evaluate **semantic relevance and domain expertise**, not just exact keyword matches. Consider real-world hiring scenarios where domain knowledge and transferable skills are highly valuable.
- Use **realistic hiring logic**: If a resume **exceeds** the JD requirements (e.g., more skills, higher education, deeper experience), the match_pct should be high — even **90-100%**.
- **Accurate scoring approach**: Recognize strong candidates appropriately. Use 85-100% for excellent matches with strong domain alignment. Use 70-84% for good matches with minor gaps. Use 50-69% for moderate fit requiring some training.
- Focus on **domain expertise and capability depth** rather than missing specific tools that can be learned.
- NEVER hallucinate or infer information not explicitly present in either document.
- NEVER nest objects inside any field — your output must remain a **flat JSON**.
- Explanations must be **insightful, honest, and actionable** — written as if advising a hiring manager on real gaps and strengths.
- Do NOT use escape characters like \\_ for underscores - use plain underscores instead.
- Do NOT escape quotes unnecessarily - only escape when required by JSON syntax.
- Do **not** include any commentary or text outside the JSON.

### Field Matching Logic:

1. **Skills**
   - **Excellent Match (85-100%)**: Resume demonstrates all required skills OR has strong domain expertise (e.g., "Deep Learning + Generative AI + LLMs" covers "TensorFlow/PyTorch" requirements)
   - **Good Match (70-84%)**: Resume covers most core requirements with minor gaps in specific frameworks that are easily learnable
   - **Moderate Match (50-69%)**: Resume shows foundational skills but missing several key requirements
   - **Weak Match (0-49%)**: Significant skill gaps requiring extensive training
   - **Key insight**: Domain expertise (ML/AI/GenAI) is more valuable than specific tool knowledge

2. **Education**
   - **Excellent Match (90-100%)**: Education level meets or exceeds requirements (B.Tech CS = 95%, B.Tech CS + Post-grad = 100%)
   - **Good Match (75-89%)**: Relevant field with strong foundational overlap (e.g., CS vs EE for ML roles)
   - **Moderate Match (60-74%)**: Different field but with relevant experience compensation
   - **Weak Match (0-59%)**: Significant education gaps not compensated by experience
   - **Key insight**: Additional relevant degrees (Data Science, ML) should boost scores significantly

3. **Experience**
   - **Excellent Match (85-100%)**: Experience duration and domain directly align (3+ years ML/AI = 90%+, hands-on GenAI = 95%+)
   - **Good Match (70-84%)**: Experience meets most requirements with minor gaps in specific technologies
   - **Moderate Match (50-69%)**: Relevant experience but with notable gaps in domain or duration
   - **Weak Match (0-49%)**: Limited relevant experience requiring significant onboarding
   - **Key insight**: Hands-on ML/AI/GenAI experience is extremely valuable and should score very highly

4. **Job Role**
   - **Excellent Match (90-100%)**: Current/recent role directly matches OR has high domain overlap (Data Scientist with ML/AI = 90%+)
   - **Good Match (75-89%)**: Adjacent role with good transferability (Software Engineer with ML experience)
   - **Moderate Match (60-74%)**: Related role but different focus area requiring some transition
   - **Weak Match (0-59%)**: Different domain or significant role mismatch
   - **Key insight**: "Data Scientist" with ML/AI focus is highly compatible with "ML Engineer/GenAI Engineer"

5. **OverallMatchPercentage**
   - Calculate as weighted average: Skills (30%), Experience (35%), Job Role (20%), Education (15%)
   - Apply **accurate hiring standards**: 85-100% = Excellent fit (strong domain expertise), 70-84% = Good fit with minor gaps, 55-69% = Potential fit with training, <55% = Poor fit
   - **Other Information** may provide minor adjustment (±3%)
   - **Key insight**: Strong candidates with domain expertise should score 85%+ even if missing some specific tools

6. **AI_Generated_Estimate_Percentage**
   - Evaluate based on: unnatural language patterns, excessive keyword stuffing, generic achievements, repetitive phrasing, lack of specific details, overly perfect formatting
   - **0-20%**: Clearly human-written with natural language and specific details
   - **21-40%**: Mostly human with possible AI assistance for formatting/phrasing
   - **41-60%**: Mixed human-AI with some generic sections
   - **61-80%**: Likely AI-generated with human oversight
   - **81-100%**: Almost certainly AI-generated with generic, templated content

### Output Format (strict):

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

Each field below is populated from the JSON files. Compare them **semantically and intelligently** using the structure below.

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
            
            result = json.loads(json_content)
            print(f"[SUCCESS] Successfully processed {resume_filename}")
            return result
        else:
            print(f"[ERROR] No valid JSON found in response for {resume_filename}")
            return {resume_filename: {"error": "No valid JSON found in LLM response", "raw_response": content}}
            
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing failed for {resume_filename}: {e}")
        return {resume_filename: {"error": f"JSON parsing failed: {str(e)}", "raw_response": content}}
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

# === Save Results ===
os.makedirs("output", exist_ok=True)
output_file = "output/resume_jd_matches_direct.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n[INFO] Results saved to: {output_file}")

# Print summary
print(f"\n[INFO] Match Summary:")
for resume_name, data in results.items():
    if "error" not in data and "OverallMatchPercentage" in data:
        overall_match = data.get("OverallMatchPercentage", 0)
        print(f"  {resume_name}: {overall_match}% overall match")
    elif "error" in data:
        print(f"  {resume_name}: ERROR - {data['error']}")

print(f"\nDirect JSON comparison completed successfully!")
