import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------
# Load documents
# -----------------------------
documents = []

DATA_PATH = "data"

print("Current Working Directory:")
print(os.getcwd())

if not os.path.exists(DATA_PATH):
    print(f"ERROR: '{DATA_PATH}' folder not found!")
    exit()

print("\nFiles found in data folder:")
print(os.listdir(DATA_PATH))

for file in os.listdir(DATA_PATH):

    if file.endswith(".txt"):

        file_path = os.path.join(DATA_PATH, file)

        with open(file_path, "r", encoding="utf-8") as f:

            text = f.read().strip()

            if text:

                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": file}
                    )
                )

                print(f"Loaded: {file}")

            else:
                print(f"WARNING: {file} is empty")

print("\nTotal Documents Loaded:", len(documents))

if len(documents) == 0:
    print("\nERROR: No documents loaded.")
    print("Check that your data folder contains .txt files with content.")
    exit()

# -----------------------------
# Split documents
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

print("Total Chunks Created:", len(docs))

if len(docs) == 0:
    print("ERROR: No chunks created.")
    exit()

# -----------------------------
# Embeddings
# -----------------------------
print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")

# -----------------------------
# Create Vector Store
# -----------------------------
print("\nCreating FAISS vector database...")

vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

vectorstore.save_local("vectorstore")

print("\nVector Database Created Successfully!")
print("Saved to: vectorstore/")