import os
from chromadb import PersistentClient

# === JD DB Path ===
CHROMA_DIR = "../chroma_db_jd"

def inspect_jd_collections(debug=False):
    """
    Inspect and print the full content of all documents in all JD ChromaDB collections.

    Args:
        debug (bool): If True, prints extra debug information.
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

    print(f"\n Found {len(collections)} JD collection(s) in '{CHROMA_DIR}':\n")

    if not collections:
        print("No collections found.")
        return

    for collection in collections:
        print(f" Collection Name: {collection.name}")
        try:
            col = client.get_collection(name=collection.name)
            count = col.count()
            print(f" - Total documents: {count}")

            if count == 0:
                print("   (Collection is empty)")
            else:
                # Fetch all documents at once
                all_docs = col.get(limit=count)
                docs = all_docs.get("documents", [])
                metadatas = all_docs.get("metadatas", [])
                ids = all_docs.get("ids", [])

                if not docs:
                    print("   (No documents found)")
                else:
                    for i, doc in enumerate(docs):
                        print(f"\n   → Document #{i+1}:")
                        print(f"      ID: {ids[i]}")
                        print(f"      Content: {doc}")
                        if debug:
                            print(f"      Metadata: {metadatas[i]}")
                    print("\n" + "-" * 50)

        except Exception as e:
            print(f" [ERROR] Could not access collection '{collection.name}': {e}")

if __name__ == "__main__":
    print("=== Inspecting Job Description Collections (Full Documents) ===")
    inspect_jd_collections(debug=True)  # Set debug=True to see metadata
    print("=== Done ===\n")
