import os
from chromadb import PersistentClient

# === Resume DB Path ===
CHROMA_DIR = "../chroma_db_resume"  # Change this path if your DB is elsewhere

def inspect_resume_collections(debug=False, sample_size=10):
    """
    Inspect and print summary of all collections in the ChromaDB resume directory.

    Args:
        debug (bool): If True, prints extra debug information.
        sample_size (int): Number of sample documents to peek from each collection.
    """
    if not os.path.exists(CHROMA_DIR):
        print(f"[ERROR] Folder '{CHROMA_DIR}' does not exist.")
        return

    try:
        client = PersistentClient(path=CHROMA_DIR)
        collections = client.list_collections()
    except Exception as e:
        print(f"[ERROR] Failed to connect to ChromaDB: {e}")
        return

    print(f"\n📁 Found {len(collections)} resume collection(s) in '{CHROMA_DIR}':\n")

    if not collections:
        print("No collections found.")
        return

    for collection in collections:
        print(f"📌 Collection Name: {collection.name}")
        try:
            col = client.get_collection(name=collection.name)
            count = col.count()
            print(f" - Total documents: {count}")

            if count == 0:
                print("   (Collection is empty)")
            else:
                # Always show up to sample_size documents, but never more than actual count
                peek_count = min(sample_size, count)
                sample = col.peek(peek_count)
                docs = sample.get("documents", [])
                metadatas = sample.get("metadatas", [])
                if not docs:
                    print("   (No documents to sample)")
                else:
                    for i, doc in enumerate(docs):
                        # Print field name from metadata if available
                        field_name = metadatas[i].get("field") if i < len(metadatas) else "unknown"
                        print(f"   → Sample #{i+1} [{field_name}]: {doc[:150]}{'...' if len(doc) > 150 else ''}")
                        if debug:
                            print(f"      [DEBUG] Full document: {doc}")
                    if debug:
                        print(f"      [DEBUG] Sample metadata: {metadatas}")

        except Exception as e:
            print(f" [ERROR] Could not access collection '{collection.name}': {e}")

        print("-" * 50)

if __name__ == "__main__":
    print("=== Inspecting Resume Collections ===")
    # Set debug=True for more details, or increase sample_size as needed
    inspect_resume_collections(debug=False, sample_size=10)
    print("=== Done ===\n")
