import os
import json
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# === Config ===
RESUME_EXTRACTION_DIR = "../resume_json"
CHROMA_DB_BASE_DIR = "../chroma_db_resumes"  # Base directory for all resume embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_single_resume(file_path, filename):
    """Load and process a single resume JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    flat_text = "\n".join([
        f"Skill: {', '.join(data.get('skill', []))}",
        f"Education: {', '.join(data.get('education', []))}",
        f"Experience: {', '.join(data.get('experience', []))}",
        f"Job Role: {', '.join(data.get('job role', []))}",
        f"Other Info: {', '.join(data.get('other information', []))}"
    ])

    metadata = {
        "source": filename,
        "job_role": data.get("job role", [""])[0] if data.get("job role") else "",
        # Store key information as strings for ChromaDB compatibility
        "skills_count": len(data.get("skill", [])),
        "education_count": len(data.get("education", [])),
        "experience_count": len(data.get("experience", [])),
        "has_other_info": len(data.get("other information", [])) > 0
    }

    return Document(page_content=flat_text, metadata=metadata), data

def embed_and_store_single_resume(doc, resume_name, base_dir, original_data):
    """Create embeddings for a single resume in its own folder"""
    # Create a clean folder name from the resume filename
    clean_name = resume_name.replace(".json", "").replace(" ", "_").replace(".", "_")
    resume_db_dir = os.path.join(base_dir, clean_name)

    # Create the directory if it doesn't exist
    os.makedirs(resume_db_dir, exist_ok=True)

    # Save the original JSON data for later use in comparisons
    json_file_path = os.path.join(resume_db_dir, "original_data.json")
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(original_data, f, indent=2, ensure_ascii=False)

    # Create embeddings
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=[doc],
        embedding=embedding,
        persist_directory=resume_db_dir
    )

    print(f"✅ Stored resume '{resume_name}' embeddings to: {resume_db_dir}")
    return resume_db_dir

def process_all_resumes(json_dir, base_dir):
    """Process all resumes and create individual embedding folders"""
    if not os.path.exists(json_dir):
        print(f"❌ Resume directory not found: {json_dir}")
        return []

    # Create base directory
    os.makedirs(base_dir, exist_ok=True)

    processed_resumes = []
    resume_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

    if not resume_files:
        print("⚠️ No resume JSON files found.")
        return []

    print(f"📁 Processing {len(resume_files)} resume files...")

    for filename in resume_files:
        file_path = os.path.join(json_dir, filename)

        try:
            # Load the resume
            resume_doc, original_data = load_single_resume(file_path, filename)

            # Create embeddings in separate folder
            db_path = embed_and_store_single_resume(resume_doc, filename, base_dir, original_data)

            processed_resumes.append({
                "filename": filename,
                "db_path": db_path,
                "document": resume_doc,
                "original_data": original_data
            })

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print(f"\n✅ Successfully processed {len(processed_resumes)} resumes")
    print(f"📂 Embedding folders created in: {base_dir}")

    return processed_resumes

# Legacy function for backward compatibility
def load_extracted_resumes(json_dir):
    """Legacy function - loads all resumes into a single list"""
    resume_docs = []
    for filename in os.listdir(json_dir):
        if not filename.endswith(".json"):
            continue
        file_path = os.path.join(json_dir, filename)
        resume_doc, _ = load_single_resume(file_path, filename)  # Ignore original_data
        resume_docs.append(resume_doc)
    return resume_docs

def embed_and_store_resumes(docs, db_dir):
    """Legacy function for backward compatibility"""
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=db_dir)
    print(f"✅ Stored {len(docs)} resume embeddings to ChromaDB at {db_dir}")

if __name__ == "__main__":
    print("🚀 Starting individual resume embedding process...")
    processed_resumes = process_all_resumes(RESUME_EXTRACTION_DIR, CHROMA_DB_BASE_DIR)

    if processed_resumes:
        print(f"\n📊 Summary:")
        for resume_info in processed_resumes:
            print(f"  - {resume_info['filename']} → {resume_info['db_path']}")
    else:
        print("❌ No resumes were processed successfully.")
