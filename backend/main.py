from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import tempfile
import shutil

# =========================
# App
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Globals (light startup)
# =========================
DB_PATH = "vectordb"
vector_db = None
embeddings = None

# =========================
# Groq Client (lightweight)
# =========================
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# =========================
# Lazy Embeddings Loader
# =========================
def get_embeddings():
    global embeddings

    if embeddings is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    return embeddings

# =========================
# Lazy Vector DB Loader
# =========================
def get_db():
    global vector_db

    if vector_db is None and os.path.exists(DB_PATH):
        from langchain_community.vectorstores import Chroma

        vector_db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=get_embeddings()
        )

    return vector_db

# =========================
# Request Model
# =========================
class QuestionRequest(BaseModel):
    question: str

# =========================
# Home
# =========================
@app.get("/")
def home():
    return {"message": "RAG Backend Running"}

# =========================
# Upload Files
# =========================
@app.post("/upload-multiple")
async def upload_multiple(files: list[UploadFile] = File(...)):
    global vector_db

    from langchain_community.document_loaders import (
        TextLoader,
        PyPDFLoader
    )
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter
    )
    from langchain_community.vectorstores import Chroma

    all_docs = []

    for file in files:
        suffix = os.path.splitext(file.filename)[1].lower()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        try:
            if suffix == ".txt":
                loader = TextLoader(
                    temp_path,
                    encoding="utf-8"
                )

            elif suffix == ".pdf":
                loader = PyPDFLoader(temp_path)

            else:
                os.remove(temp_path)
                continue

            docs = loader.load()
            all_docs.extend(docs)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if not all_docs:
        return {
            "message": "No valid files uploaded",
            "chunks": 0
        }

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(all_docs)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
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
    db = get_db()

    if db is None:
        return {
            "answer": "Please upload files first.",
            "sources": []
        }

    query = data.question

    docs = db.similarity_search(query, k=3)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

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
            {
                "role": "user",
                "content": prompt
            }
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

    return {
        "message": "Knowledge base cleared"
    }