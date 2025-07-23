import os
import json
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

JD_EXTRACTION_DIR = "../JD_extraction"
CHROMA_DB_DIR = "../chroma_db_jd"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_extracted_jds(json_dir):
    jd_docs = []
    for filename in os.listdir(json_dir):
        if not filename.endswith(".json"):
            continue
        file_path = os.path.join(json_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        doc_id = os.path.splitext(filename)[0]
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
        }

        jd_docs.append(Document(page_content=flat_text, metadata=metadata))
    return jd_docs

def embed_and_store_jds(docs, db_dir):
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=db_dir)
    print(f"✅ Stored {len(docs)} JD embeddings to ChromaDB at {db_dir}")

if __name__ == "__main__":
    jd_documents = load_extracted_jds(JD_EXTRACTION_DIR)
    if not jd_documents:
        print("⚠️ No JD JSON documents found.")
    else:
        embed_and_store_jds(jd_documents, CHROMA_DB_DIR)
