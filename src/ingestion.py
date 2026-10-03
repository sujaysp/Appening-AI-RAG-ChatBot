from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    PDF_PATH,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)


def load_and_chunk_pdf(pdf_path: Path):
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    print(f"Loaded {len(documents)} pages from PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    for chunk in chunks:
        page_number = chunk.metadata.get("page", 0) + 1
        chunk.metadata["page"] = page_number
        chunk.metadata["source"] = pdf_path.name

    return chunks


def create_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = pc.list_indexes().names()

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        print("Pinecone index created.")
    else:
        print(f"Pinecone index already exists: {PINECONE_INDEX_NAME}")

    return pc


def ingest_documents():
    chunks = load_and_chunk_pdf(PDF_PATH)

    create_pinecone_index()

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY,
    )

    print("Uploading embeddings to Pinecone...")

    vector_store.add_documents(
        documents=chunks,
        namespace="agentic-ai",
    )

    print("Ingestion completed successfully.")

    return len(chunks)