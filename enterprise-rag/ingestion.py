# ingestion.py

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from docx import Document

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


# ===================================
# Load PDF
# ===================================

def load_pdf(file_path):

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    return documents


# ===================================
# Load TXT
# ===================================

def load_txt(file_path):

    loader = TextLoader(
        file_path,
        encoding="utf-8"
    )

    documents = loader.load()

    return documents


# ===================================
# Load DOCX
# ===================================

def load_docx(file_path):

    doc = Document(file_path)

    text = "\n".join(
        para.text
        for para in doc.paragraphs
    )

    return [{
        "page_content": text,
        "metadata": {
            "source": file_path
        }
    }]


# ===================================
# Load Web Page
# ===================================

def load_webpage(url):

    loader = WebBaseLoader(url)

    documents = loader.load()

    return documents


# ===================================
# Split Documents
# ===================================

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(
        documents
    )

    return chunks


# ===================================
# Generic Loader
# ===================================

def load_document(source):

    source = source.strip()

    if source.startswith("http"):

        documents = load_webpage(source)

    elif source.endswith(".pdf"):

        documents = load_pdf(source)

    elif source.endswith(".txt"):

        documents = load_txt(source)

    elif source.endswith(".docx"):

        documents = load_docx(source)

    else:

        raise ValueError(
            "Unsupported document type."
        )

    return documents


# ===================================
# Ingestion Pipeline
# ===================================

def ingest_document(source):

    documents = load_document(source)

    chunks = split_documents(
        documents
    )

    return chunks


# ===================================
# Test
# ===================================

if __name__ == "__main__":

    source = input(
        "Enter file path or URL: "
    )

    chunks = ingest_document(
        source
    )

    print(
        f"\nGenerated {len(chunks)} chunks"
    )

    print("\nFirst Chunk:")
    print("-" * 50)

    print(
        chunks[0].page_content[:500]
    )