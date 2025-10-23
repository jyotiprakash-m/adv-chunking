"""
agentic_chunker_v2.py

A clean, typed, and modular refactor of the original AgenticChunker.

Features
- Groups short "propositions" into semantic chunks using an LLM.
- Generates and updates compact summaries and generalized titles.
- Returns LangChain Document objects for each chunk (page_content + metadata).
- Allows injecting a custom LLM (ChatOpenAI) or passing OpenAI API key.
- Deterministic truncated chunk IDs for compact references.

Requires:
- langchain-core (for Document, prompts)
- langchain-openai (for ChatOpenAI)
- langchain (for create_extraction_chain_pydantic)
- dotenv (optional) if loading API key from .env
"""

from __future__ import annotations

import os
import uuid
import re
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()


class AgenticChunker2:
    """
    AgenticChunker
    ----------------
    LLM-powered semantic clustering of short statements ("propositions").

    Usage:
        ac = AgenticChunker(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))
        ac.add_propositions(["The month is October.", "The year is 2023."])
        ac.pretty_print_chunks()
        docs = ac.get_chunks_as_documents()
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        id_truncate_limit: int = 5,
        generate_new_metadata: bool = True,
        print_logging: bool = True,
        llm: Optional[ChatOpenAI] = None,
    ):
        """
        Parameters
        ----------
        openai_api_key: Optional[str]
            OpenAI API key (will fallback to env var OPENAI_API_KEY)
        model: str
            Model name to pass to ChatOpenAI
        temperature: float
            LLM temperature
        id_truncate_limit: int
            length of truncated chunk ids
        generate_new_metadata: bool
            whether to regenerate summaries/titles when new propositions are added
        print_logging: bool
            toggles console prints
        llm: Optional[ChatOpenAI]
            If provided, this LLM instance will be used (overrides openai_api_key + model)
        """
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")

        if openai_api_key is None and llm is None:
            raise ValueError("OpenAI API key not provided and no LLM instance supplied")

        self.llm = llm or ChatOpenAI(model=model, temperature=temperature)

        self.chunks: Dict[str, Dict[str, Any]] = {}
        self.id_truncate_limit = int(id_truncate_limit)
        self.generate_new_metadata = bool(generate_new_metadata)
        self.print_logging = bool(print_logging)

    # ----------------------------
    # Public API
    # ----------------------------
    def add_propositions(self, propositions: List[str]) -> None:
        """Add multiple propositions (in order)."""
        for p in propositions:
            self.add_proposition(p)

    def add_proposition(self, proposition: str) -> None:
        """Add a single proposition to an existing chunk or create a new chunk."""
        if self.print_logging:
            print(f"\nAdding proposition: '{proposition}'")

        # If no chunks exist, create first chunk
        if len(self.chunks) == 0:
            if self.print_logging:
                print("No existing chunks — creating a new chunk.")
            self._create_new_chunk(proposition)
            return

        # Try to find a relevant chunk
        chunk_id = self._find_relevant_chunk(proposition)

        if chunk_id:
            if self.print_logging:
                print(f"Found chunk {chunk_id} — appending proposition.")
            self._add_to_chunk(chunk_id, proposition)
        else:
            if self.print_logging:
                print("No relevant chunk found — creating a new chunk.")
            self._create_new_chunk(proposition)

    def get_chunks(self) -> Dict[str, Dict[str, Any]]:
        """Return internal chunk dictionary (raw)."""
        return self.chunks

    def get_chunks_as_documents(self) -> List[Document]:
        """
        Return chunks as a list of LangChain Documents.
        Each chunk becomes a single Document with metadata.
        """
        docs: List[Document] = []
        for chunk_id, ch in self.chunks.items():
            page_content = "\n".join(ch["propositions"])
            metadata = {
                "chunk_id": chunk_id,
                "title": ch.get("title"),
                "summary": ch.get("summary"),
                "chunk_index": ch.get("chunk_index"),
                "proposition_count": len(ch.get("propositions", [])),
            }
            docs.append(Document(page_content=page_content, metadata=metadata))
        return docs

    def pretty_print_chunks(self) -> None:
        """Nicely print full chunk details for debugging."""
        print(f"\nYou have {len(self.chunks)} chunks\n")
        for chunk_id, chunk in sorted(self.chunks.items(), key=lambda kv: kv[1].get("chunk_index", 0)):
            print(f"Chunk #{chunk['chunk_index']} — ID: {chunk_id}")
            print(f"Title: {chunk.get('title')}")
            print(f"Summary: {chunk.get('summary')}")
            print("Propositions:")
            for prop in chunk["propositions"]:
                print(f"  - {prop}")
            print("")

    def pretty_print_chunk_outline(self) -> None:
        """Print a compact outline of chunk titles + summaries."""
        print("Chunk Outline\n")
        for chunk_id, chunk in sorted(self.chunks.items(), key=lambda kv: kv[1].get("chunk_index", 0)):
            print(f"Chunk ({chunk_id}): {chunk.get('title')}")
            print(f"Summary: {chunk.get('summary')}\n")

    # ----------------------------
    # Internal Helpers
    # ----------------------------
    def _create_new_chunk(self, proposition: str) -> str:
        """Create a new chunk containing the given proposition and return its id."""
        new_chunk_id = str(uuid.uuid4())[: self.id_truncate_limit]
        summary = self._get_new_chunk_summary(proposition)
        title = self._get_new_chunk_title(summary)

        self.chunks[new_chunk_id] = {
            "chunk_id": new_chunk_id,
            "propositions": [proposition],
            "title": title,
            "summary": summary,
            "chunk_index": len(self.chunks),
        }

        if self.print_logging:
            print(f"Created new chunk ({new_chunk_id}): {title}")

        return new_chunk_id

    def _add_to_chunk(self, chunk_id: str, proposition: str) -> None:
        """Add proposition and optionally update metadata."""
        chunk = self.chunks[chunk_id]
        chunk["propositions"].append(proposition)

        if self.generate_new_metadata:
            # Generate updated summary and title based on updated propositions
            chunk["summary"] = self._update_chunk_summary(chunk["propositions"], chunk.get("summary"))
            chunk["title"] = self._update_chunk_title(chunk["propositions"], chunk.get("summary"), chunk.get("title"))

    # ----------------------------
    # LLM-Prompted Operations
    # ----------------------------
    def _get_new_chunk_summary(self, proposition: str) -> str:
        """Ask the LLM to create a new chunk summary for a single proposition (one-sentence)."""
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
Generate a very brief 1-sentence summary that explains what this chunk would be about. Generalize when appropriate (e.g., 'apples' -> 'food').
Only respond with the summary string, nothing else.
""",
                ),
                ("user", "Determine the summary of the new chunk that this proposition will go into:\n{proposition}"),
            ]
        )

        runnable = PROMPT | self.llm
        out = runnable.invoke({"proposition": proposition}).content
        return str(out).strip()

    def _get_new_chunk_title(self, summary: str) -> str:
        """Ask the LLM to create a short title for a chunk summary (few words)."""
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the steward of a group of chunks. Given a short summary, respond with a concise chunk title (a few words) that generalizes the summary.
Only respond with the title, nothing else.
""",
                ),
                ("user", "Determine the title of the chunk that this summary belongs to:\n{summary}"),
            ]
        )

        runnable = PROMPT | self.llm
        out = runnable.invoke({"summary": summary}).content
        return str(out).strip()

    def _update_chunk_summary(self, propositions: List[str], current_summary: Optional[str] = None) -> str:
        """Given a group's propositions and optional current summary, return an updated 1-sentence summary."""
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the steward of a group of propositions. A new proposition was added; create a brief 1-sentence summary that captures what this chunk is about.
Generalize when appropriate. Only respond with the updated summary.
""",
                ),
                ("user", "Chunk's propositions:\n{proposition}\n\nCurrent chunk summary:\n{current_summary}"),
            ]
        )

        runnable = PROMPT | self.llm
        joined = "\n".join(propositions)
        out = runnable.invoke({"proposition": joined, "current_summary": current_summary or ""}).content
        return str(out).strip()

    def _update_chunk_title(self, propositions: List[str], current_summary: Optional[str], current_title: Optional[str]) -> str:
        """Given a group's propositions and current summary/title, return an updated short title."""
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the steward of a group of propositions. A new proposition was added; produce a concise title (a few words) that best represents the chunk.
Generalize when appropriate. Only respond with the updated title.
""",
                ),
                (
                    "user",
                    "Chunk's propositions:\n{proposition}\n\nChunk summary:\n{current_summary}\n\nCurrent chunk title:\n{current_title}",
                ),
            ]
        )

        runnable = PROMPT | self.llm
        joined = "\n".join(propositions)
        out = runnable.invoke({"proposition": joined, "current_summary": current_summary or "", "current_title": current_title or ""}).content
        return str(out).strip()

    # ----------------------------
    # Chunk Matching (which chunk does proposition belong to?)
    # ----------------------------
    def _find_relevant_chunk(self, proposition: str) -> Optional[str]:
        """
        Ask LLM to choose an existing chunk ID for a proposition or return 'No chunks'.
        We use a small pydantic schema to extract the chunk_id if the LLM returns it verbosely.
        """
        current_chunk_outline = self._get_chunk_outline_text()

        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
Determine whether or not the given proposition should belong to any of the existing chunks.
If it should belong to a chunk, return the chunk id (exactly). If it should not belong to any chunk, return "No chunks".
Only output the chunk id or the phrase "No chunks".
""",
                ),
                ("user", "Current Chunks:\n--Start of current chunks--\n{current_chunk_outline}\n--End of current chunks--"),
                ("user", "Determine if the following statement should belong to one of the chunks outlined:\n{proposition}"),
            ]
        )

        runnable = PROMPT | self.llm
        raw_out = runnable.invoke({"current_chunk_outline": current_chunk_outline, "proposition": proposition}).content
        raw_out = str(raw_out).strip()

        # Use structured output parsing instead of deprecated extraction chain
        class ChunkID(BaseModel):
            chunk_id: Optional[str]

        parser = PydanticOutputParser(pydantic_object=ChunkID)

        try:
            # Try to parse the response as structured output
            parsed_response = parser.parse(raw_out)
            chunk_found = parsed_response.chunk_id
        except:
            # Fallback: extract chunk_id from raw text if parsing fails
            # Create regex pattern for chunk ID (alphanumeric, specific length)
            pattern = rf'\b[a-zA-Z0-9]{{{self.id_truncate_limit}}}\b'
            chunk_id_match = re.search(pattern, raw_out)
            chunk_found = chunk_id_match.group(0) if chunk_id_match else None

        # Normalize common "No chunks" variants
        if chunk_found and chunk_found.lower().startswith("no chunk"):
            return None

        # Validate chunk id length — it must match self.id_truncate_limit
        if chunk_found and len(chunk_found) != self.id_truncate_limit:
            return None

        # Ensure chunk exists
        if chunk_found and chunk_found not in self.chunks:
            return None

        return chunk_found

    def _get_chunk_outline_text(self) -> str:
        """Create a textual outline of existing chunks for LLM prompts."""
        parts = []
        for chunk_id, chunk in self.chunks.items():
            parts.append(f"- Chunk ID: {chunk_id}\n  - Chunk Name: {chunk.get('title')}\n  - Chunk Summary: {chunk.get('summary')}\n")
        return "\n".join(parts)
