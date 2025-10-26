""" Healthcare Assistant API Router """
import asyncio
import json
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, status, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from agents import Runner, trace, gen_trace_id, Agent, OpenAIChatCompletionsModel, WebSearchTool, ModelSettings, function_tool
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import requests
import tempfile

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
    text: str
    pages: int
    confidence: float

class HealthManager:
    """Manages health-related operations."""

    async def run(self, request: HealthcareRequest):
        """Process the uploaded file and send summary to the email."""
        logger.info(request)

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
            pages: Optional[list] = None
        ) -> str:
            """
            Extracts text from an image or PDF using the OpenTyphoon OCR API.
            """
            url = os.getenv('OPEN_ROUTER_OCR_BASE_URL')
            
            if not url:
                raise ValueError("OPEN_ROUTER_OCR_BASE_URL is not set in environment variables.")

            with open(image_path, 'rb') as file:
                files = {'file': file}
                data = {
                    'task_type': task_type,
                    'max_tokens': str(max_tokens),
                    'temperature': str(temperature),
                    'top_p': str(top_p),
                    'repetition_penalty': str(repetition_penalty)
                }

                if pages:
                    data['pages'] = json.dumps(pages)
                
                
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
                            logger.error(f"Error processing {page_result.get('filename', 'unknown')}: {page_result.get('error', 'Unknown error')}")
                    logger.info("OCR extraction completed. %d pages processed.", len(extracted_texts))
                    return '\n'.join(extracted_texts)

                else:
                    logger.error(f"Error: {response.status_code} - {response.text}")
                    return f"Error: {response.status_code} - {response.text}"

        OCR_AGENT_INSTRUCTIONS = """
        You are an OCR agent specialized in extracting text from medical documents.
        Your task is to accurately extract and return the text content from the provided document.
        Ensure that the extracted text is clear and well-formatted for further analysis.
        """
        
        ocr_agent = Agent(
            name="OCR Extractor",
            instructions=OCR_AGENT_INSTRUCTIONS,
            tools=[extract_text_from_file],
            model="gpt-4o-mini",
            output_type=OCRResponse,
        )

        # Trace and Execute
        trace_id = gen_trace_id()
        with trace("HealthCare Analysis", trace_id=trace_id):
            logger.info("Starting healthcare analysis workflow.")
            trace_message = f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}"
            logger.info(trace_message)

            logger.info("Performing OCR on the uploaded file.")
            extracted_text = await self.perform_ocr(request.file, ocr_agent)

            logger.info("Analyzing medical data.")
            medical_summary = await self.analyze_medical_data(extracted_text)
            logger.info("Medical data analyzed.")

            logger.info("Scheduling appointment.")
            appointment_details = await self.schedule_appointment(request.email)
            summary_email_content = f"{medical_summary}\n\n{appointment_details}"
            logger.info("Appointment scheduled. %s", appointment_details)

            return summary_email_content

    async def perform_ocr(self, file: UploadFile, ocr_agent: Agent) -> str:
        """Perform OCR on the uploaded file and return extracted text."""
        filename = file.filename or ""
        ext = os.path.splitext(filename)[-1] or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            file_path = tmp.name
            contents = await file.read()
            tmp.write(contents)

        try:
            run_result = await Runner.run(ocr_agent, file_path)

            if isinstance(run_result, OCRResponse):
                result = run_result.text
            elif isinstance(run_result, dict) and "text" in run_result:
                if isinstance(run_result["text"], bytes):
                    result = run_result["text"].decode("utf-8", errors="ignore")
                else:
                    result = str(run_result["text"])
            elif isinstance(run_result, bytes):
                result = run_result.decode("utf-8", errors="ignore")
            else:
                result = str(run_result)

        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

        return result


    async def analyze_medical_data(self, text: str) -> str:
        """Analyze the extracted medical data and generate a summary."""
        # Simulate medical data analysis
        logger.info(text)
        logger.info("Analyzing text of length %d", len(text))
        await asyncio.sleep(1)
        return "Medical summary based on the extracted data."

    async def schedule_appointment(self, email: str) -> str:
        """Schedule an appointment and return confirmation details."""
        # Simulate appointment scheduling
        await asyncio.sleep(1)
        return "Appointment scheduled for next week."


@router.post("/run", response_model=Dict[str, Any])
async def run_healthcare_assistant(request: HealthcareRequest) -> Dict[str, Any]:
    health_manager = HealthManager()
    try:
        result = await health_manager.run(request)
    except Exception as e:
        logger.error(f"Health analysis error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error")
    return {"result": result}