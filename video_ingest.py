# video_ingest.py
import os
import whisper
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

VIDEO_PATH = "uploaded_video.mp4"   # user jo video upload kare
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "video_collection"  # alag collection PDF se

def transcribe_video(video_path):
    print("🎬 Video se audio + text nikal raha hun...")
    print("(Pehli baar Whisper model download hoga ~140MB, thoda waqt lagega)")
    
    # Whisper model load karo (base = balanced, small = tez, medium = accurate)
    model = whisper.load_model("base")
    
    # Transcribe karo
    result = model.transcribe(video_path, language=None)  # auto-detect language
    script = result["text"]
    
    print(f"✅ Script nikli: {len(script)} characters")
    return script


def save_to_vectorstore(script):
    print("✂️  Chunks bana raha hun...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # chhota rakha taake names miss na hon
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    chunks = splitter.split_text(script)
    print(f"✅ Total chunks: {len(chunks)}")
    
    print("🧠 Embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    print("💾 Chroma DB mein save...")
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )
    print("🎉 Done! Video script vector DB mein save ho gayi.")


if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video nahi mila: {VIDEO_PATH}")
        print("Video ko 'uploaded_video.mp4' naam se folder mein rakho.")
    else:
        script = transcribe_video(VIDEO_PATH)
        
        # Script ko file mein bhi save karo (reference ke liye)
        with open("video_script.txt", "w", encoding="utf-8") as f:
            f.write(script)
        print("📄 Script 'video_script.txt' mein bhi save ho gayi.")
        
        save_to_vectorstore(script)