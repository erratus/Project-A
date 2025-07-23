import os
import json
import re
import requests
import fitz  
from docx import Document   
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import dotenv

# Ensure the environment variables are loaded
dotenv.load_dotenv()
HF_TOKEN = os.getenv("hf_token")

class LLMResumeParser:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        load_dotenv()
        self.model = model_name
        self.client = InferenceClient(model=self.model, token=HF_TOKEN)
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self):
        return '''
 You are an expert resume parser.

Your task is to extract structured data from unstructured resume text into exactly five predefined flat sections. You must return only one well-formed JSON object using the exact keys and definitions below. Do not add, omit, rename, or modify any section.

1. "skill"
Include only hard technical skills such as programming languages, frameworks, libraries, software tools, platforms, cloud services, or technical methodologies.
 Do NOT include certifications, soft skills, role descriptions, degrees, or company names.
 Examples: Python, TensorFlow, SQL, AWS, Docker, CI/CD, Agile, Power BI

2. "education"
Include only formal academic qualifications such as:

Schooling (10th, 12th)

Undergraduate (e.g., B.Tech, BSc)

Postgraduate (e.g., M.Tech, MBA, MSc)

Doctorate (PhD)

Each item must contain only the degree/program and institution/school/college name.
 Do NOT include years, CGPA, certifications, online courses, bootcamps, or platforms like Coursera.
 Examples:

B.Tech in Computer Science from IIT Bombay

10th from Delhi Public School

MBA from IIM Ahmedabad

3. "experience"
Each experience must be written as a single plain text string. Do not return objects or structured fields. Combine job title, company, location, and responsibilities into a single string, separated by punctuation.

Example:
"Data Scientist at ABM, Nov22–Present — Developed SQL AI Agent using LangChain and Streamlit for natural language to SQL query conversion."

Do not use key-value pairs like { "Job title": ..., "Company": ... }
Do not return arrays of objects or nested fields.

4. "job role"
Include only one specific job title that best represents the candidate’s most recent or primary designation.
 Do NOT include multiple titles or composite roles.
 Example:

Data Scientist

5. "other information"
Include all remaining information that does not fit the above four sections, such as:

Certifications

Soft skills

Languages spoken

Hobbies and interests

Awards and achievements

Relocation or career objectives

Extracurriculars or personal statements

Certifications from any provider (e.g., AWS, Coursera, Google) should be included here.

 Output Format
Return exactly this structure:

json
Copy
Edit
{
  "skill": [],
  "education": [],
  "experience": [],
  "job role": [],
  "other information": []
}
 Strict Rules
Each section must be a flat list of plain text strings.

Do NOT return nested JSON or arrays of objects.

Do NOT infer or hallucinate data not explicitly in the text.

Do NOT wrap the JSON in an array or include any other explanation.

Do NOT include the prompt or label the JSON.

Replace all newlines (\n, \\n) and tab characters (\t, \\t) with \\n and \\t inside values.

All five keys must always be present — even if their lists are empty.

Each value must appear in only one section.
Do NOT return nested JSON or key-value objects inside any section — each entry must be a plain string.
"job role" must contain only one title — if more are present, choose the most representative one.
'''

    def clean_text(self, text: str) -> str:
        # Clean and normalize text for LLM input
        text = text.replace('\n', '. ').replace('\r', '')
        text = text.replace('\\', ' or ')
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'[\u200b-\u206f\u2e00-\u2e7f]', '', text)
        return re.sub(' +', ' ', text).strip()

    def extract_fields(self, resume_text: str) -> dict:
        cleaned_text = self.clean_text(resume_text)
        
        try:
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": cleaned_text}
            ]
            
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=None
            )
            
            # Extract the content from the response
            raw_output = response.choices[0].message.content.strip()

            print("\n Raw LLM Output:\n", raw_output)

            json_start = raw_output.find('{')
            json_end = raw_output.rfind('}')
            if json_start == -1 or json_end == -1:
                print(" Could not find valid JSON object in the LLM response.")
                return {}

            json_clean = raw_output[json_start:json_end+1]

            try:
                result = json.loads(json_clean)
            except json.JSONDecodeError as e:
                print(" JSON decoding failed:", e)
                return {}

            # Ensure all required keys are present
            required_keys = ["skill", "education", "experience", "job role", "other information"]
            for key in required_keys:
                if key not in result or not isinstance(result[key], list):
                    result[key] = []

            return result

        except Exception as e:
            print(f" Error calling or parsing LLM output: {e}")
            return {}

    def save_to_json(self, data: dict, output_dir: str, original_file: str):
        if not data or all(not v for v in data.values()):
            print(" Skipping save: No valid JSON returned.")
            return

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(original_file))[0]
        output_path = os.path.join(output_dir, f"{base_name}.json")

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f" Saved: {output_path}")
        except Exception as e:
            print(f" Failed to save {output_path}: {e}")

    def extract_text_from_file(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            try:
                doc = fitz.open(file_path)
                texts = []
                for page in doc:
                    text = page.get_text()
                    if not text.strip():
                        blocks = page.get_text("blocks")
                        text = "\n".join(
                            b[4].strip() for b in sorted(blocks, key=lambda b: (b[1], b[0])) if b[4].strip()
                        )
                    texts.append(text.strip())
                return " ".join(texts)
            except Exception as e:
                print(f" Error reading PDF {file_path}: {e}")
                return ""
        elif ext == ".docx":
            try:
                doc = Document(file_path)
                return " ".join(para.text.strip() for para in doc.paragraphs if para.text.strip())
            except Exception as e:
                print(f" Error reading DOCX {file_path}: {e}")
                return ""
        else:
            print(f"Unsupported file format: {file_path}")
            return ""


#  Clear old JSON files  
def clear_json_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                try:
                    os.remove(file_path)
                    print(f" Deleted old JSON: {file_path}")
                except Exception as e:
                    print(f" Could not remove file {file_path}: {e}")
    else:
        os.makedirs(folder_path)


#  Main resume parsing logic
def process_resumes(input_path: str, output_dir: str):
    parser = LLMResumeParser()

    clear_json_folder(output_dir)

    if os.path.isfile(input_path):
        files = [input_path] if input_path.lower().endswith((".pdf", ".docx")) else []
    elif os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                 if f.lower().endswith((".pdf", ".docx"))]
    else:
        print(f" Invalid path: {input_path}")
        return

    for file_path in files:
        print(f"\n Processing: {file_path}")
        text = parser.extract_text_from_file(file_path)
        if not text.strip():
            print(f" Skipped empty or unreadable file: {file_path}")
            continue
        parsed = parser.extract_fields(text)
        parser.save_to_json(parsed, output_dir, file_path)


# Uncomment this block to run directly
if __name__ == "__main__":
    input_path = "../resumes"
    output_path = "../resume_json"
    process_resumes(input_path, output_path)
 