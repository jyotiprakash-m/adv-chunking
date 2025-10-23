from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language
)
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from typing import List
import tempfile, os, re


# Import your request model
from models.request_response import RequestChunk

#  custom chunkers
from utils import SlidingWindowSplitter, AgenticChunker2

router = APIRouter(prefix="/chunk", tags=["Chunking"])

# ======================
# Helper functions
# ======================

def get_length_function(name: str):
    """Return a custom length function for chunking."""
    def word_count_length(text): return len(text.split())
    def simple_sentence_count(text): return len([s for s in text.split('.') if s.strip()])
    def advanced_sentence_count(text):
        abbreviations = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'etc.', 'i.e.', 'e.g.', 'vs.']
        for i, abbr in enumerate(abbreviations):
            text = text.replace(abbr, f'ABBR{i}')
        sentences = re.split(r'[.!?]+', text)
        return len([s.strip() for s in sentences if s.strip()])
    
    if name == "word":
        return word_count_length
    elif name == "simple_sentence_count":
        return simple_sentence_count
    elif name == "advanced_sentence_count":
        return advanced_sentence_count
    else:
        return None


def extract_text_from_pdf(uploaded_file: UploadFile) -> str:
    """Extract text from an uploaded PDF file."""
    filename = getattr(uploaded_file, "filename", None)
    if not filename or not isinstance(filename, str) or not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.file.read())
        tmp_path = tmp.name
    
    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        text = "\n".join([d.page_content for d in docs])
    finally:
        os.remove(tmp_path)
    
    return text

# ======================
# Main route
# ======================

@router.post("")
async def chunk_pdf(
    file: UploadFile = File(...),
    method: str = Form("block"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    strip_whitespace: bool = Form(True),
    language: str = Form(None),
    separator: str = Form(None),
    is_separator_regex: bool = Form(False),
    length_function: str = Form(None),
    window_size: int = Form(None),
    step_size: int = Form(None),
    granularity: str = Form(None),
    model: str = Form(None),
    temperature: float = Form(0.0)
):
    """
    POST /chunk
    Upload a PDF file, extract text, and perform chunking.
    """
    text = extract_text_from_pdf(file)
    docs: List[Document] = []

    # 1️⃣ Recursive Block Splitter
    if method == "block":
        if language:
            try:
                lang_enum = getattr(Language, language.upper())
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=lang_enum,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=[separator] if separator else None,
                    is_separator_regex=is_separator_regex,
                    strip_whitespace=strip_whitespace
                )
            except AttributeError:
                raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=[separator] if separator else None,
                is_separator_regex=is_separator_regex,
                strip_whitespace=strip_whitespace
            )
        docs = splitter.create_documents([text])

    # 2️⃣ Sentence-Based Splitter
    elif method == "sentence":
        length_func = get_length_function(length_function)
        if language:
            try:
                lang_enum = getattr(Language, language.upper())
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=lang_enum,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=length_func,
                    strip_whitespace=strip_whitespace
                )
            except AttributeError:
                raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=length_func,
                strip_whitespace=strip_whitespace
            )
        docs = splitter.create_documents([text])

    # 3️⃣ Sliding Window Splitter
    elif method == "sliding_window":
        if not window_size or not step_size:
            raise HTTPException(status_code=400, detail="window_size and step_size are required for sliding_window method")
        splitter = SlidingWindowSplitter(
            window_size=window_size,
            step_size=step_size,
            granularity=granularity or "sentence",
        )
        docs = splitter.create_documents([text])

    # 4️⃣ Agentic Chunker (LLM-based)
    elif method == "agentic":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY in environment.")
        ac = AgenticChunker2(
            openai_api_key=key,
            model=model or "gpt-4o-mini",
            temperature=temperature or 0.0,
            print_logging=False
        )
        ac.add_propositions([text])
        docs = ac.get_chunks_as_documents()

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported chunking method: {method}")

    # ✅ Prepare JSON response
    response_data = [
        {"chunk_id": i + 1, "content": d.page_content, "metadata": d.metadata}
        for i, d in enumerate(docs)
    ]
    return JSONResponse(content={"num_chunks": len(docs), "chunks": response_data})
