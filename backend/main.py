from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
import tempfile
import shutil
import os

# =========================
# FastAPI App
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later replace with frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Global Variables
# =========================
DB_PATH = "vectordb"
vector_db = None

# =========================
# Groq Client
# =========================
client = Groq(
    api_key=os.getenv("YOUR_REAL_KEY")
)

# =========================
# Embeddings Model
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# Load Existing DB If Exists
# =========================
if os.path.exists(DB_PATH):
    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

# =========================
# Request Model
# =========================
class QuestionRequest(BaseModel):
    question: str

# =========================
# Home Route
# =========================
@app.get("/")
def home():
    return {"message": "RAG Backend Running"}

# =========================
# Upload Multiple Files
# =========================
@app.post("/upload-multiple")
async def upload_multiple(files: list[UploadFile] = File(...)):
    global vector_db

    all_docs = []

    for file in files:
        suffix = os.path.splitext(file.filename)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        # TXT Loader
        if suffix == ".txt":
            loader = TextLoader(temp_path, encoding="utf-8")

        # PDF Loader
        elif suffix == ".pdf":
            loader = PyPDFLoader(temp_path)

        else:
            continue

        docs = loader.load()
        all_docs.extend(docs)

        os.remove(temp_path)

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(all_docs)

    # Create vector DB
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    vector_db.persist()

    return {
        "message": "Files indexed successfully",
        "chunks": len(chunks)
    }

# =========================
# Ask Question
# =========================
@app.post("/ask")
async def ask(data: QuestionRequest):
    global vector_db

    if vector_db is None:
        return {"answer": "Please upload files first.", "sources": []}

    query = data.question

    docs = vector_db.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    sources = list(set([
        doc.metadata.get("source", "Unknown")
        for doc in docs
    ]))

    prompt = f"""
Answer only using the context below.
If answer is not found, say:
Not found in uploaded files.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": sources
    }

# =========================
# Clear DB
# =========================
@app.delete("/clear")
def clear_db():
    global vector_db

    vector_db = None

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    return {"message": "Knowledge base cleared"}