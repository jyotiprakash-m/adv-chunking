"""
Career Center Router

Provides FastAPI endpoints for interacting with the "Me" chatbot
that answers questions about Jyoti Prakash Mohanta. The chatbot uses
OpenAI's tool calling to record user details and unknown questions via
Pushover notifications.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from openai import OpenAI
from pydantic import BaseModel, Field
from pypdf import PdfReader

# Attempt to load environment variables as early as possible
load_dotenv(override=True)

router = APIRouter(prefix="/career", tags=["Career Center"])


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def push(text: str) -> None:
    """Send a notification through Pushover.

    Raises a RuntimeError if the request fails so callers can handle it.
    """

    token = os.getenv("PUSHOVER_TOKEN")
    user = os.getenv("PUSHOVER_USER")

    if not token or not user:
        raise RuntimeError("Pushover credentials are not configured.")

    response = requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": token,
            "user": user,
            "message": text,
        },
        timeout=10,
    )
    response.raise_for_status()


def record_user_details(email: str, name: str = "Name not provided", notes: str = "not provided") -> Dict[str, str]:
    """Record that a user wants to get in touch."""

    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question: str) -> Dict[str, str]:
    """Record a question that could not be answered."""

    push(f"Recording {question}")
    return {"recorded": "ok"}


# Tool schemas used by OpenAI tool calling
record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user",
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it",
            },
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context",
            },
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered",
            }
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

TOOLS = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
]


class Me:
    """Encapsulates chatbot state and behaviour."""

    def __init__(self) -> None:
        self.openai = OpenAI()
        self.name = "Jyoti Prakash Mohanta"

        base_path = Path(__file__).resolve().parent.parent
        me_dir = base_path / "utils"

        # Load LinkedIn content from PDF
        linkedin_pdf = me_dir / "jpm.pdf"
        if not linkedin_pdf.exists():
            raise FileNotFoundError(f"LinkedIn PDF not found at {linkedin_pdf}")

        reader = PdfReader(str(linkedin_pdf))
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text

        # Load summary text
        summary_file = me_dir / "summary.txt"
        if not summary_file.exists():
            raise FileNotFoundError(f"Summary file not found at {summary_file}")

        self.summary = summary_file.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Tool handling
    # ------------------------------------------------------------------
    @staticmethod
    def handle_tool_call(tool_calls: Any) -> List[Dict[str, Any]]:
        """Execute tool calls returned by OpenAI and format responses."""

        results: List[Dict[str, Any]] = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool = globals().get(tool_name)
            if tool is None:
                raise RuntimeError(f"Unknown tool requested: {tool_name}")
            tool_result = tool(**arguments)
            results.append(
                {
                    "role": "tool",
                    "content": json.dumps(tool_result),
                    "tool_call_id": tool_call.id,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Prompt construction and chat handling
    # ------------------------------------------------------------------
    def system_prompt(self) -> str:
        """Build the system prompt with profile context."""

        prompt = (
            f"You are acting as {self.name}. You are answering questions on {self.name}'s website, "
            f"particularly questions related to {self.name}'s career, background, skills and experience. "
            f"Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. "
            f"You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. "
            f"Be professional and engaging, as if talking to a potential client or future employer who came across the website. "
            f"If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. "
            f"If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "
        )

        prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return prompt

    def chat(self, message: str, history: List[Dict[str, Any]]) -> str:
        """Run a chat completion loop handling tool calls as needed."""

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt()}
        ] + history + [
            {"role": "user", "content": message}
        ]

        while True:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,  # type: ignore
                tools=TOOLS, # type: ignore
            )

            choice = response.choices[0]
            finish_reason = choice.finish_reason

            if finish_reason == "tool_calls":
                message_obj = choice.message
                tool_calls = message_obj.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message_obj.model_dump())
                messages.extend(results)
                continue

            if finish_reason == "stop":
                return choice.message.content or ""

            raise RuntimeError(f"Unexpected finish reason: {finish_reason}")


# Instantiate the chatbot at import time so it can be reused across requests
ME = Me()


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """Represents a single message in the conversation history."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Request payload for the chat endpoint."""

    message: str = Field(..., description="The latest user message to process.")
    history: List[ChatMessage] = Field(
        default_factory=list,
        description="Conversation history as a list of messages.",
    )


class ChatResponse(BaseModel):
    """Response payload containing the assistant reply."""

    role: str = Field("assistant", description="The role generating the response.")
    content: str = Field(..., description="Assistant message content.")


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """Interact with the Me chatbot via FastAPI."""

    try:
        response_text = await run_in_threadpool(
            ME.chat,
            payload.message,
            [message.model_dump() for message in payload.history],
        )
    except Exception as exc:  # pragma: no cover - unexpected errors
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(role="assistant", content=response_text)



