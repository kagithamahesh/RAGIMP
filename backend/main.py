from fastapi import FastAPI,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.embeddings import HuggingFaceEmbeddings

import os
import shutil
from transformers import pipeline
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"]
)

#paths 
UPLOAD_DIR = "uploads"
VECTOR_DIR = "vectordb"

os.makedirs(UPLOAD_DIR,exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

#Models
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
     max_new_tokens=220,
    do_sample=True,
    temperature=0.7
)

#Load Or Create DB

db = Chroma(
    persist_directory=VECTOR_DIR,
    embedding_function=embeddings
)

#Request Model
class AskRequest(BaseModel):
    question:str



#helpers

def load_document(file_path,ext):
    if ext =="txt":
        loader = TextLoader(file_path,encoding="utf-8")
    else:
        loader =PyPDFLoader(file_path)

    docs =loader.load() 

    return docs

#Health check

@app.get("/")
def home():
    return {"message":"RAG Backend running"}

@app.post("/upload-multiple")
async def upload_multiple(files:list[UploadFile]=File(...)):
    all_docs = []
    for file in files:
        save_path = os.path.join(UPLOAD_DIR,file.filename)

        with open(save_path,"wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        ext = file.filename.split(".")[-1].lower()
        docs =load_document(save_path,ext=ext)

        for d in docs:
            d.metadata["source"] = file.filename
        
        all_docs.extend(docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(all_docs)
        if len(chunks) == 0:
            return {"message": "No readable text found"}
         
         # recreate persistent db
        global db
        db= Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=VECTOR_DIR
        )

        return{
            "message" :"files indexed successfully",
            "files":len(files),
            "chunks":len(chunks)
        }
    
# ask qusation
@app.post("/ask")
async def ask(req: AskRequest):

    global db

    query = req.question

    docs = db.similarity_search(query, k=3)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Use the context below to answer clearly.

Context:
{context}

Question:
{query}

Answer:
"""

    result = llm(prompt,
                 max_new_tokens=120,
                do_sample=True,
                temperature=0.4,
                return_full_text=False)

    answer = result[0]["generated_text"].strip()

    # citations
    sources = []

    for d in docs:
        src = d.metadata.get("source", "Unknown")
        if src not in sources:
            sources.append(src)

    return {
        "answer": answer,
        "sources": sources
    }

# @app.post("/ask") 
# async def ask(req:AskRequest):
#     global db
#     query = req.question
#     docs = db.similarity_search(query,k=3)

#     context = "\n\n".join([d.page_content for d in docs])

#     prompt = f"""
# Use the content below to answer clearly.

# Context:
# {context}

# Question:
# {query}

# Answer:
# """
#     result = llm(prompt)
#     answer = result[0]["generated_text"]

#     #citations
#     sources = []

#     for d in docs:
#             src = d.metadata.get("source","Unknown")

#             if src not in sources:
#                 sources.append(src)

#             return{
#                     "answer": answer,
#                 "sources": sources
#             }     
        
#  #clear DB

@app.delete("/clear")
def clear_db():
    global db

    if os.path.exists(VECTOR_DIR):
        shutil.rmtree(VECTOR_DIR)

    os.makedirs(VECTOR_DIR,exist_ok=True)

    db = Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=embeddings
    )      

    return {"message": "Knowledge base cleared"}