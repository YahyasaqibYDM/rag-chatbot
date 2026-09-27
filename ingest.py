# ingest.py
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PDF_PATH = "document.pdf"
CHROMA_DIR = "./chroma_db"

def main():
    print("📖 PDF padh raha hun...")
    reader = PdfReader(PDF_PATH)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    print(f"✅ Total characters: {len(text)}")

    print("✂️  Chunks bana raha hun...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    print(f"✅ Total chunks: {len(chunks)}")

    print("🧠 Embeddings bana raha hun (pehli baar model download hoga ~80MB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("💾 Chroma DB mein save kar raha hun...")
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="pdf_collection"
    )
    print("🎉 Done! Ab 'python app.py' chalao.")

if __name__ == "__main__":
    main()