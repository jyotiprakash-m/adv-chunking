import re
from typing import Callable, List, Optional, Dict, Any, Union
from langchain_core.documents import Document

try:
    import tiktoken  # optional for token-based granularity
except ImportError:
    tiktoken = None


class SlidingWindowSplitter:
    """
    A flexible sliding-window text splitter for LangChain documents.

    Features:
    ----------
    • Supports granularity: character, word, sentence, or token.
    • Allows custom length functions (for counting chars, words, tokens, etc.)
    • Adds rich metadata to each chunk (index, overlap %, sizes, etc.)
    • Produces LangChain `Document` objects.
    • Fully deterministic & reproducible.

    Example:
    --------
    >>> splitter = SlidingWindowSplitter(
    ...     window_size=200,
    ...     step_size=100,
    ...     granularity="word"
    ... )
    >>> docs = splitter.create_documents(["This is a long text that will be chunked..."])
    >>> print(len(docs), "chunks created")
    """

    def __init__(
        self,
        window_size: int = 1000,
        step_size: int = 500,
        granularity: str = "character",
        length_function: Optional[Callable[[str], int]] = None,
        strip_whitespace: bool = True,
        tokenizer_name: str = "cl100k_base",  # for token granularity
    ):
        if step_size <= 0 or window_size <= 0:
            raise ValueError("Both `window_size` and `step_size` must be positive integers.")

        self.window_size = window_size
        self.step_size = step_size
        self.granularity = granularity.lower().strip()
        self.strip_whitespace = strip_whitespace
        self.length_function = length_function or len

        # Tokenizer for token-based mode
        if self.granularity == "token":
            if not tiktoken:
                raise ImportError("`tiktoken` is required for token-based granularity. Install with `pip install tiktoken`.")
            self.tokenizer = tiktoken.get_encoding(tokenizer_name)

    # -------------------------------------------
    # TEXT UNIT EXTRACTION & REASSEMBLY
    # -------------------------------------------
    def _get_text_units(self, text: str) -> List[Any]:
        """Split text into discrete units based on granularity."""
        if self.granularity == "character":
            return list(text)
        elif self.granularity == "word":
            return text.split()
        elif self.granularity == "sentence":
            sentences = re.split(r"(?<=[.!?])\s+", text)
            return [s.strip() for s in sentences if s.strip()]
        elif self.granularity == "token":
            tokens = self.tokenizer.encode(text)
            return tokens
        else:
            raise ValueError("Granularity must be 'character', 'word', 'sentence', or 'token'.")

    def _units_to_text(self, units: List[Any]) -> str:
        """Recombine units into a string."""
        if self.granularity == "character":
            return "".join(units)
        elif self.granularity == "word":
            return " ".join(units)
        elif self.granularity == "sentence":
            return " ".join(units)
        elif self.granularity == "token":
            return self.tokenizer.decode(units)
        return ""

    # -------------------------------------------
    # CORE SLIDING LOGIC
    # -------------------------------------------
    def split_text(self, text: str) -> List[str]:
        """Split a text into overlapping chunks."""
        if not text or not text.strip():
            return []

        # Optimized path for simple character-based case
        if self.granularity == "character" and self.length_function == len:
            chunks = []
            for i in range(0, len(text), self.step_size):
                chunk = text[i : i + self.window_size]
                if self.strip_whitespace:
                    chunk = chunk.strip()
                if chunk:
                    chunks.append(chunk)
                if i + self.window_size >= len(text):
                    break
            return chunks

        # General path (word, sentence, token, or custom length)
        units = self._get_text_units(text)
        chunks = []

        for i in range(0, len(units), self.step_size):
            window_units = units[i : i + self.window_size]
            chunk = self._units_to_text(window_units)
            if self.strip_whitespace:
                chunk = chunk.strip()

            if chunk and self._measure_length(chunk) > 0:
                chunks.append(chunk)

            if i + self.window_size >= len(units):
                break

        return chunks

    def _measure_length(self, text: str) -> int:
        """Compute chunk length via custom or default function."""
        try:
            return int(self.length_function(text))
        except Exception:
            return len(text)

    # -------------------------------------------
    # DOCUMENT CREATION
    # -------------------------------------------
    def create_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Document]:
        """Split multiple texts and return LangChain Documents."""
        documents = []
        metadatas = metadatas or [{} for _ in texts]

        for idx, text in enumerate(texts):
            chunks = self.split_text(text)
            meta = metadatas[idx] if idx < len(metadatas) else {}

            for chunk_idx, chunk in enumerate(chunks):
                overlap = max(0, self.window_size - self.step_size)
                doc_metadata = {
                    **meta,
                    "text_index": idx,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "granularity": self.granularity,
                    "window_size": self.window_size,
                    "step_size": self.step_size,
                    "chunk_length": self._measure_length(chunk),
                    "overlap": overlap,
                    "overlap_percentage": round((overlap / self.window_size) * 100, 2),
                }
                documents.append(Document(page_content=chunk, metadata=doc_metadata))

        return documents

    # -------------------------------------------
    # DEBUG & VISUALIZATION
    # -------------------------------------------
    def preview(self, text: str, n: int = 3) -> None:
        """Print the first few chunks for inspection."""
        chunks = self.split_text(text)
        for i, chunk in enumerate(chunks[:n]):
            print(f"\n--- Chunk {i+1}/{len(chunks)} ---")
            print(chunk)
            print(f"[Length: {self._measure_length(chunk)} | {self.granularity}-based]")

