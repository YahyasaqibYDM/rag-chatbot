# app.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import google.generativeai as genai

load_dotenv()

CHROMA_DIR = "./chroma_db"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY .env file mein set nahi hai!")

print(f"🔑 API Key mili: {GEMINI_API_KEY[:10]}...")
genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ====== Embeddings ======
print("🔄 Loading embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# ====== Vector Stores — STARTUP PAR KHALI ======
# Dono None honge. Sirf user upload karne par banenge.
pdf_store = None
video_store = None

print("ℹ️  PDF store: khaali (user upload karega)")
print("ℹ️  Video store: khaali (user upload karega)")

# ====== Gemini ======
print("🔄 Initializing Gemini...")
gemini_model = genai.GenerativeModel("gemini-3.8-flash")
print("✅ Ready! User ab PDF/Video upload kar sakta hai.")


class Query(BaseModel):
    question: str
    source: str = "pdf"


@app.post("/ask")
def ask(query: Query):
    try:
        if query.source == "video":
            store = video_store
            source_name = "video"
        else:
            store = pdf_store
            source_name = "PDF document"

        # Agar user ne abhi tak upload nahi kiya
        if store is None:
            return {
                "answer": f"⚠️ Abhi tak koi {source_name} upload nahi hui.\n\n"
                          f"Barah-e-karam pehle input bar ke upar "
                          f"{'📄 PDF' if query.source == 'pdf' else '🎬 Video'} "
                          f"button se file upload karein, phir sawal poochein.",
                "sources": []
            }

        # Retrieval
        docs = store.similarity_search(query.question, k=8)

        seen = set()
        unique_docs = []
        for d in docs:
            if d.page_content not in seen:
                seen.add(d.page_content)
                unique_docs.append(d)
        docs = unique_docs[:6]

        if not docs:
            return {
                "answer": f"Mujhe {source_name} mein is sawal ka jawab nahi mila. "
                          f"Kuch aur poochne ki koshish karein.",
                "sources": []
            }

        context = "\n\n---\n\n".join(d.page_content for d in docs)

        prompt = f"""You are a professional AI assistant. Answer the user's question based on the context provided below.

RULES:
1. Carefully read ALL the context before answering.
2. If the question asks about people, places, or things mentioned in the context, describe them professionally.
3. Combine information from multiple parts of the context if needed.
4. Only if the information is COMPLETELY absent from the context, say:
   "I could not find this information in the provided {source_name}."
5. Answer in a clear, professional, and helpful tone.
6. Use bullet points or sections if the answer is long.
7. If you spot spelling variants of the same name, treat them as the same.

Context:
{context}

Question: {query.question}

Professional Answer:"""

        response = gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 1000,
            }
        )

        return {"answer": response.text, "sources": []}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"answer": f"⚠️ Error: {str(e)}", "sources": []}


# ====== PDF Upload ======
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        global pdf_store

        save_path = "uploaded_document.pdf"
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"✅ PDF save: {save_path}")

        print("📖 PDF padh raha hun...")
        reader = PdfReader(save_path)
        text = "\n".join((page.extract_text() or "") for page in reader.pages)

        if not text.strip():
            return {"message": "⚠️ PDF mein koi text nahi mila (shayad scanned images hain)."}

        print(f"✅ Text nikla: {len(text)} chars")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=150
        )
        chunks = splitter.split_text(text)

        if not chunks:
            return {"message": "⚠️ PDF se koi chunks nahi bane."}

        # Purani collection delete karo (agar hai)
        try:
            if pdf_store is not None:
                pdf_store.delete_collection()
        except Exception:
            pass

        pdf_store = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
            collection_name="pdf_collection"
        )

        return {
            "message": f"✅ PDF process ho gayi! {len(chunks)} chunks bane.",
            "preview": text[:300] + ("..." if len(text) > 300 else "")
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"message": f"⚠️ Error: {str(e)}"}


# ====== Video Upload ======
@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    try:
        global video_store

        save_path = "uploaded_video.mp4"
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"✅ Video save: {save_path}")

        import whisper
        print("🎬 Whisper se script nikal raha hun... (2-5 min)")
        model = whisper.load_model("base")
        result = model.transcribe(save_path)
        script = result["text"]
        print(f"✅ Script: {len(script)} chars")

        with open("video_script.txt", "w", encoding="utf-8") as f:
            f.write(script)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )
        chunks = splitter.split_text(script)

        if not chunks:
            return {"message": "⚠️ Video mein koi speech nahi mili."}

        try:
            if video_store is not None:
                video_store.delete_collection()
        except Exception:
            pass

        video_store = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
            collection_name="video_collection"
        )

        return {
            "message": f"✅ Video process ho gayi! {len(chunks)} chunks bane.",
            "preview": script[:300] + ("..." if len(script) > 300 else "")
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"message": f"⚠️ Error: {str(e)}"}


@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()