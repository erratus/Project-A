import os
import json
import re
import fitz  # PyMuPDF
from docx import Document
from huggingface_hub import InferenceClient
import dotenv

dotenv.load_dotenv()
HF_TOKEN = os.getenv("hf_token")
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

class LLMJDParser:
    def __init__(self, model_name=MODEL_NAME):
        self.model = model_name
        
        self.client = InferenceClient(model=self.model, token=HF_TOKEN)
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self):
        return '''
You are an expert job description parser.

Your task is to extract structured data from unstructured job description (JD) text into exactly five predefined flat sections. You must return only one well-formed JSON object using the exact keys and rules provided below. Do not add, omit, rename, or modify any section.

 Section Definitions
1. "skill"
Only include hard technical or professional skills, including tools, programming languages, platforms, frameworks, software, or methodologies that the job requires or mentions.

Do NOT include certifications, soft skills, job roles, company names, or generic traits.

Examples:
Python, TensorFlow, Agile, Jira, AWS, CI/CD, SQL, Kubernetes, Snowflake

2. "education"
Include only formal education qualifications expected or required for the job, such as degrees, universities, academic programs, or certifications from recognized platforms (e.g. Coursera, edX, Udemy).

 Do NOT include training from non-academic sources unless clearly stated as an educational course.

 Examples:
Bachelor’s degree in Computer Science,
MBA from a reputed institute,
Coursera Machine Learning Specialization

3. "experience"
Include only descriptions of responsibilities, work or project experience, achievements, years of experience, or qualifications the employer expects the candidate to have.

Focus on what the candidate is expected to do or bring to the role.

 Examples:
3+ years of experience in data analysis,
Experience building scalable ML models,
Proficient in designing REST APIs

4. "job role"
Include only one clear job title or position mentioned in the JD. If multiple titles are mentioned, choose the primary one.

 Do NOT list multiple titles or variations. Return only one, clean designation.

 Examples:
Machine Learning Engineer,
Data Scientist,
Cloud Solutions Architect

5. "other information"
Include all remaining relevant content not covered by the above sections, such as:

Soft skills (e.g., communication, leadership)

Language requirements

Work culture or company values

Employment type (e.g., full-time, remote)

Relocation, travel, or shift requirements

Perks and benefits

Visa requirements

Any additional notes or instructions

Examples:
Excellent verbal and written communication skills,
Hybrid work model,
Flexible working hours and wellness benefits

Output Format
Return exactly this JSON object:

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
 Global Rules
Each section must be a flat list of plain text strings.

Do NOT return nested JSON or arrays of key-value objects.

Do NOT include summaries or inferred content.

Do NOT wrap the JSON in an outer array.

Replace all line breaks (\n, \\n) and tabs (\t, \\t) with \\n and \\t inside string values.

Always include all five keys, even if their lists are empty.

Each item must appear in only one section.

job role must contain only one job title — no composite or repeated roles.

 Output must be a single valid JSON object. Do not include this prompt or any additional explanation in the output.

'''

    def clean_text(self, text: str) -> str:
        text = text.replace('\n', '. ').replace('\r', '')
        text = text.replace('\\', ' or ')
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'[\u200b-\u206f\u2e00-\u2e7f]', '', text)
        return re.sub(' +', ' ', text).strip()

    def extract_fields(self, jd_text: str) -> dict:
        cleaned_text = self.clean_text(jd_text)

        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": cleaned_text}
            ]

            response = self.client.chat_completion(
                messages=messages,
                max_tokens=1024
            )

            raw_output = response.choices[0].message.content.strip()

            print("\n🪵 Raw LLM Output:\n", raw_output)

            json_start = raw_output.find('{')
            json_end = raw_output.rfind('}')
            if json_start == -1 or json_end == -1:
                print(" Could not find valid JSON object in the LLM response.")
                return {}

            json_clean = raw_output[json_start:json_end + 1]

            try:
                result = json.loads(json_clean)
            except json.JSONDecodeError as e:
                print(" JSON decoding failed:", e)
                return {}

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
        elif ext == ".txt":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                print(f" Error reading TXT {file_path}: {e}")
                return ""
        else:
            print(f"Unsupported file format: {file_path}")
            return ""


# Clear old JSON files
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


#  Main JD parsing logic
def process_jds(input_path: str, output_dir: str):
    parser = LLMJDParser()

    clear_json_folder(output_dir)

    if os.path.isfile(input_path):
        files = [input_path] if input_path.lower().endswith((".pdf", ".docx", ".txt")) else []
    elif os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                 if f.lower().endswith((".pdf", ".docx", ".txt"))]
    else:
        print(f" Invalid path: {input_path}")
        return

    for file_path in files:
        print(f"\n Processing JD: {file_path}")
        text = parser.extract_text_from_file(file_path)
        if not text.strip():
            print(f" Skipped empty or unreadable JD file: {file_path}")
            continue
        parsed = parser.extract_fields(text)
        parser.save_to_json(parsed, output_dir, file_path)


# Uncomment this block to run directly
if __name__ == "__main__":
    input_path = "../JD"
    output_path = "../JD_extraction"
    process_jds(input_path, output_path)
