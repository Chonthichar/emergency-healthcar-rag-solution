import os
import json
import shutil
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

print("🚀 Starting the FOLDER-BASED indexing process...")

# --- 1. Configuration ---
load_dotenv()
DB_LOCATION = os.getenv("DATABASE_LOCATION")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
TOPICS_JSON_PATH = 'data/topics.json'
TOPICS_BASE_PATH = "data/topics/"

# --- 2. Initialize Components ---
print("Initializing models and text splitter...")
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# --- 3. Clean Up Old Database ---
if os.path.exists(DB_LOCATION):
    print(f"🧹 Removing old database at {DB_LOCATION}")
    shutil.rmtree(DB_LOCATION)

# --- 4. Load topics.json ---
print(f"📚 Loading topic map from '{TOPICS_JSON_PATH}'...")
try:
    with open(TOPICS_JSON_PATH, 'r', encoding='utf-8') as f:
        topic_map = json.load(f)
except FileNotFoundError:
    print(f"❌ ERROR: Could not find topics map at {TOPICS_JSON_PATH}")
    exit()

# --- 5. Process Each Topic Folder ---
print("📁 Processing each topic folder and all .md files within...")
all_docs = []
processed_topics_count = 0

# Get the list of directories in the base path
topic_folders = [d for d in os.listdir(TOPICS_BASE_PATH) if os.path.isdir(os.path.join(TOPICS_BASE_PATH, d))]

for folder_name in topic_folders:
    # Use the folder name as the key to look up in topics.json
    if folder_name in topic_map:
        topic_id = topic_map[folder_name]
        topic_folder_path = os.path.join(TOPICS_BASE_PATH, folder_name)
        processed_topics_count += 1

        # Find all .md files within this specific topic folder
        md_files_in_folder = [f for f in os.listdir(topic_folder_path) if f.endswith('.md')]

        if not md_files_in_folder:
            print(f"  - 🟡 NOTE: No .md files found in folder '{folder_name}'")
            continue

        for md_file in md_files_in_folder:
            file_path = os.path.join(topic_folder_path, md_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                chunks = text_splitter.split_text(content)

                for chunk in chunks:
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source_file": md_file,
                            "topic_name": folder_name, # The topic name is the folder name
                            "topic_id": topic_id
                        }
                    )
                    all_docs.append(doc)
            except Exception as e:
                print(f"  - ❌ ERROR processing file '{file_path}': {e}")
    else:
        print(f"  - ⚠️ WARNING: Folder '{folder_name}' found on disk but not in topics.json. Skipping.")


# --- 6. Final Report ---
print(f"\n✅ Successfully processed {processed_topics_count} topic folders and created {len(all_docs)} text chunks.")

if not all_docs:
    print("\n❌ CRITICAL ERROR: No documents were indexed. Check folder names and paths.")
    exit()

# --- 7. Create and Populate the Vector Store ---
print("💾 Creating the vector store... (This may take a few minutes)")
vector_store = Chroma.from_documents(
    documents=all_docs,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory=DB_LOCATION,
)

print("\n✨ Indexing complete! Your vector database is finally ready. ✨")