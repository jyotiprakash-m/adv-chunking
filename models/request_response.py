"""
Request and Response Models for Advanced RAG Backend

This module defines Pydantic models for API request and response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Literal, Union
from datetime import datetime
from fastapi import UploadFile, File, Form


# ========================================
# FILE UPLOAD MODELS
# ========================================

class RequestChunk(BaseModel):
    """Request model for file upload and chunking"""
    
    # File input
    file: UploadFile = File(..., description="The file to be uploaded and chunked")
    
    method: Literal["block", "sentence", "sliding_window", "agentic"] = Field(
        default="block",
        description="The chunking method to use"
    )
    chunk_size: int = Field(default=1000, ge=1, le=10000, description="Maximum size of each chunk")
    chunk_overlap: int = Field(default=200, ge=0, description="Overlap between chunks")
    strip_whitespace: bool = Field(
        default=True,
        description="Whether to strip whitespace from chunks"
    )
    language: Optional[Literal[
        "CPP", "GO", "JAVA", "KOTLIN", "JS", "TS",
        "PHP", "PROTO", "PYTHON", "RST", "RUBY",
        "RUST", "SCALA", "SWIFT", "MARKDOWN", "LATEX",
        "HTML", "SOL", "CSHARP", "COBOL", "C", "LUA",
        "PERL", "HASKELL", "ELIXIR", "POWERSHELL",
        "VISUALBASIC6"
    ]] = Field(
        default=None,
        description="Language code for sentence chunking (if applicable). Accepts uppercase values."
    )
    
    # Only Block specific parameter
    separator: Optional[str] = Field(
        default=None,
        description="Custom separator for block chunking method"
    )
    is_separator_regex: Optional[bool] = Field(
        default=None,
        description="Whether the separator is a regex pattern"
    )
    
    # Sentence specific parameter
    length_function: Optional[Literal["word","sentence", "simple_sentence_count", "advanced_sentence_count"]] = Field(
        default=None,
        description="Length function to use for sentence chunking"
    )
    
    
    # Sliding window specific parameters
    window_size: Optional[str] = Field(None, ge=1, description="Window size for sliding window method")
    step_size: Optional[str] = Field(None, ge=1, description="Step size for sliding window method")
    granularity: Optional[Literal["character", "word", "sentence", "token"]] = Field(
        default=None,
        description="Granularity for chunking"
    )
    
    # Agentic specific parameters
    model: Optional[
        Union[
            Literal[
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-4",
                "gpt-4-0613",
                "gpt-4-32k",
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4o-realtime-preview"
            ],
            str
        ]
    ] = Field(
        default=None,
        description="Optional model name for the LLM to use. If omitted, a default model will be chosen by the system."
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Optional sampling temperature for generation. If omitted, the system default will be used."
    )


    
    # Metadata about the file or chunk
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    
    @field_validator('chunk_overlap')
    def validate_overlap(cls, v, info):
        """Ensure chunk_overlap is less than chunk_size"""
        if 'chunk_size' in info.data and v >= info.data['chunk_size']:
            raise ValueError('chunk_overlap must be less than chunk_size')
        return v
    
    @field_validator('step_size')
    def validate_step_size(cls, v, info):
        """Ensure step_size is valid for sliding window"""
        if v is not None and 'window_size' in info.data and info.data.get('window_size'):
            if v > info.data['window_size']:
                raise ValueError('step_size cannot be greater than window_size')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "method": "block",  # Fixed: must be one of the allowed Literal values
                "chunk_size": 500,
                "chunk_overlap": 100,
                "separator": "\n\n",
            }
        }



# ========================================
# EXPORT ALL MODELS
# ========================================

__all__ = [
    "RequestChunk",
]
