""" Healthcare Assistant API Router """
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status, UploadFile,File,Form
from pydantic import BaseModel, Field
from agents import Runner, trace, gen_trace_id, Agent,ModelSettings, function_tool
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import requests
import tempfile
from PyPDF2 import PdfReader

# Load environment variables from .env file
load_dotenv()

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/healthcare-assistant", tags=["Healthcare Assistant"])

# HealthcareRequest model
class HealthcareRequest(BaseModel):
    file: UploadFile = Field(..., description="Uploaded medical document file")
    email: str = Field(..., description="User's email address for sending the summary")
    
class OCRResponse(BaseModel):
    extracted_text: str
    confidence_score: float | None = None

class HealthManager:
    """Manages health-related operations."""

    async def run(self, file_path, email):

        # =====================
        # OCR Agent
        # =====================

        # tools

        @function_tool
        def extract_text_from_file(
            image_path: str,
            task_type: str = "default",
            max_tokens: int = 16000,
            temperature: float = 0.1,
            top_p: float = 0.6,
            repetition_penalty: float = 1.2,
            pages: Optional[List[int]] = None
        ) -> str:
            """
            Extracts text from an image or PDF using the OpenTyphoon OCR API.
            """
            url = os.getenv('OPEN_ROUTER_OCR_BASE_URL')
            
            if not url:
                raise ValueError("OPEN_ROUTER_OCR_BASE_URL is not set in environment variables.")

            # Determine pages to extract if not specified
            if pages is None:
                if image_path.lower().endswith('.pdf'):
                    reader = PdfReader(image_path)
                    num_pages = len(reader.pages)
                    pages = list(range(1, num_pages + 1))  # 1-based indexing
                else:
                    pages = [1]  # Assume single page for images

            with open(image_path, 'rb') as file:
                files = {'file': file}
                data = {
                    'task_type': task_type,
                    'max_tokens': str(max_tokens),
                    'temperature': str(temperature),
                    'top_p': str(top_p),
                    'repetition_penalty': str(repetition_penalty),
                    'pages': json.dumps(pages)  # Always send pages
                }
                
                api_key = os.getenv('OPENTYPHOON_API_KEY')
                if not api_key:
                    raise ValueError("OPENTYPHOON_API_KEY is not set in environment variables.")

                headers = {'Authorization': f'Bearer {api_key}'}
                response = requests.post(url, files=files, data=data, headers=headers)

                if response.status_code == 200:
                    result = response.json()
                    extracted_texts = []
                    for page_result in result.get('results', []):
                        if page_result.get('success') and page_result.get('message'):
                            content = page_result['message']['choices'][0]['message']['content']
                            try:
                                parsed_content = json.loads(content)
                                text = parsed_content.get('natural_text', content)
                            except json.JSONDecodeError:
                                text = content
                            extracted_texts.append(text)
                        elif not page_result.get('success'):
                            logging.error(f"Error processing {page_result.get('filename', 'unknown')}: {page_result.get('error', 'Unknown error')}")
                    logging.info("OCR extraction completed. %d pages processed.", len(extracted_texts))
                    return '\n'.join(extracted_texts)

                else:
                    logging.error(f"Error: {response.status_code} - {response.text}")
                    return f"Error: {response.status_code} - {response.text}"


  
        OCR_AGENT_INSTRUCTIONS = """
        You are an OCR agent for medical documents.
        Extract clear, well-formatted text from the document located at the given file path (URL or local path): {file_path}.
        Return structured output as defined in the OCRResponse model.
        """
        
        ocr_agent = Agent(
            name="OCR Extractor",
            instructions=OCR_AGENT_INSTRUCTIONS,
            tools=[extract_text_from_file],
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.1),  # Low temperature for accuracy
            output_type=OCRResponse,
        )

        # Trace and Execute
        trace_id = gen_trace_id()
        with trace("HealthCare Analysis", trace_id=trace_id):
            logger.info("Starting healthcare analysis workflow.")
            trace_message = f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}"
            logger.info(trace_message)

            logger.info("Performing OCR on the uploaded file.")
            # use the provided file path and email parameters (request was undefined)
            extracted_text = await self.perform_ocr(file_path, ocr_agent)

            logger.info("Analyzing medical data.")
            medical_summary = await self.analyze_medical_data(extracted_text)
            logger.info("Medical data analyzed.")

            logger.info("Scheduling appointment.")
            appointment_details = await self.schedule_appointment(email)
            summary_email_content = f"{medical_summary}\n\n{appointment_details}"
            logger.info("Appointment scheduled. %s", appointment_details)

            return summary_email_content
    async def perform_ocr(self, file_path: str, ocr_agent: Agent) -> str:
        """Perform OCR on the provided file path and return the extracted text."""
        try:
            logger.info(f"Performing OCR for file: {file_path}")
            run_result = await Runner.run(ocr_agent, file_path)
            output = getattr(run_result, "output", None)

            logger.info(f"OCR output: {output}")

            if output is None:
                raise ValueError("No output returned from OCR agent.")

            # Case 1: Structured OCRResponse
            if isinstance(output, OCRResponse):
                return output.extracted_text

            # Case 2: Dictionary output
            if isinstance(output, dict):
                text_data = (
                    output.get("extracted_text")
                    or output.get("text")
                    or output.get("data")
                )
                if text_data is None:
                    raise ValueError("No text field found in OCR output.")
                return text_data.decode("utf-8", errors="replace") if isinstance(text_data, bytes) else str(text_data)

            # Case 3: Raw bytes output
            if isinstance(output, bytes):
                return output.decode("utf-8", errors="replace")

            # Fallback: convert anything else to string
            return str(output)

        except Exception as e:
            # Log or handle exceptions cleanly
            raise RuntimeError(f"OCR processing failed for {file_path}: {e}")

    async def analyze_medical_data(self, text: str) -> str:
        """Summarize the extracted medical text using a summarization agent."""
        from agents import Agent, ModelSettings

        SUMMARIZATION_AGENT_INSTRUCTIONS = """
        You are a medical summarization agent.
        Summarize the following medical text clearly and concisely.
        Focus on diagnoses, findings, medications, and recommendations.
        Return the summary in plain text (no JSON).
        """

        summarizer_agent = Agent(
            name="Medical Summarizer",
            instructions=SUMMARIZATION_AGENT_INSTRUCTIONS,
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.3),
        )

        run_result = await Runner.run(summarizer_agent, text)
        summary = getattr(run_result, "output", None)

        if not summary:
            raise ValueError("No summary generated by summarization agent.")

        return str(summary)

    async def schedule_appointment(self, email: str) -> str:
        """Schedule an appointment and return confirmation details."""
        # Simulate appointment scheduling
        await asyncio.sleep(1)
        return "Appointment scheduled for next week."


@router.post("/run", response_model=Dict[str, Any])
async def run_healthcare_assistant(
    file: UploadFile = File(..., description="Uploaded medical document file"),
    email: str = Form(..., description="User's email address for sending the summary")
) -> Dict[str, Any]:
    health_manager = HealthManager()
    file_path: Optional[str] = None
    try:
        filename = file.filename or ""
        ext = os.path.splitext(filename)[-1] or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            file_path = tmp.name
            contents = await file.read()
            tmp.write(contents)

        # await the manager run and return its result
        result = await health_manager.run(file_path, email)

    except Exception as e:
        logger.error(f"Health analysis error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error")
    finally:
        # Clean up the temporary file if it exists
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except Exception:
                logger.warning("Failed to delete temporary file: %s", file_path)

    return {
        "status": "success",
        "summary_email_content": result,
        }