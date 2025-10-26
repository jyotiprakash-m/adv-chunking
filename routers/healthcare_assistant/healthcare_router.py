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
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import requests
import tempfile
from PyPDF2 import PdfReader
from routers.healthcare_assistant.doctors import Doctor
from utils.database import DATABASE_URL,engine
from sqlmodel import  Field, Session, select
import base64
import mimetypes
from sendgrid.helpers.mail import Attachment, FileContent, FileName, FileType, Disposition

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

        # Trace and Execute
        trace_id = gen_trace_id()
        with trace("HealthCare Analysis", trace_id=trace_id):
            logger.info("Starting healthcare analysis workflow.")
            trace_message = f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}"
            logger.info(trace_message)

            logger.info("Performing OCR on the uploaded file.")
            extracted_text = await self.perform_ocr(file_path)

            logger.info("Analyzing medical data.")
            medical_summary = await self.analyze_medical_data(extracted_text)
            logger.info("Medical data analyzed.")

            logger.info("Scheduling appointment.")
            final_email_details = await self.schedule_appointment(email, medical_summary, file_path)
            summary_email_content = f"{trace_message}\n\n{medical_summary}\n\n{final_email_details}"
            logger.info("Appointment scheduled. %s", final_email_details[:100])

            return summary_email_content
        
    def extract_text_from_file(
        self,
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

    async def perform_ocr(self, file_path: str) -> str:
        """Perform OCR on the provided file path and return the extracted text."""
        try:
            logger.info(f"Performing OCR for file: {file_path}")
            result = self.extract_text_from_file(file_path)
            logger.info(f"OCR result: {result[:100]}...")  # Log first 100 chars

        except Exception as e:
            logger.error(f"OCR processing failed for {file_path}: {e}")
            return f"OCR processing failed: {e}"

        return result

    async def analyze_medical_data(self, text: str) -> str:
        """Summarize the extracted medical text using a summarization agent."""

        logger.info("Content to be summarized: %s", text[:100])  # Log first 100 chars

        # ==========================
        # Define Medical Adviser Agent
        # ==========================
        adviser_agent_instructions = """
        You are a medical adviser agent.
        Your role is to provide expert medical insights based on provided patient or report data.
        When called, return short, clear advice in plain text focused on key observations, potential diagnoses,
        and next-step recommendations.
        """
        adviser_agent = Agent(
            name="Medical Adviser",
            instructions=adviser_agent_instructions,
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.3),
        )

        # Expose adviser as callable tool
        adviser_tool = adviser_agent.as_tool(
            tool_name="medical_adviser_tool",
            tool_description="Provides expert medical advice based on the extracted text."
        )

        # ==========================
        # Define Summarizer Agent
        # ==========================
        SUMMARIZATION_AGENT_INSTRUCTIONS = """
        You are a medical pathologist agent. who hast 15 years of experience.
        Your task is to summarize the following medical report into clear, structured notes.
        You need to suggest the specialist like cardiologist, neurologist etc based on the report in separate sections.
        Focus on diagnoses, findings, medications, and recommendations.
        You may call the `medical_adviser_tool` to obtain expert medical insights or to clarify ambiguous details.

        Follow this process:
        1. Read the text and summarize it briefly.
        2. If you are unsure about a clinical implication, call `medical_adviser_tool` with the relevant text portion.
        3. Combine your summary and the adviser's insights into a final cohesive paragraph.

        Return the final summary as plain text (no JSON).
        """

        summarizer_agent = Agent(
            name="Medical Summarizer",
            instructions=SUMMARIZATION_AGENT_INSTRUCTIONS,
            model="gpt-4o-mini",
            tools=[adviser_tool],
            model_settings=ModelSettings(temperature=0.3),
        )

        # ==========================
        # Run summarization
        # ==========================
        run_result = await Runner.run(summarizer_agent, text)

        # Try to access consistent output attribute
        summary = getattr(run_result, "final_output", None) or getattr(run_result, "output", None)

        if not summary:
            raise ValueError("No summary generated by summarization agent.")

        logger.info("Medical summary generated successfully.")
        return str(summary)

    async def schedule_appointment(self, email: str, medical_summary: str, file_path: Optional[str] = None) -> str:
        """Schedule an appointment and return confirmation details."""
        # Simulate appointment scheduling
        # Agent will find out the specialist and schedule accordingly
        
        @function_tool
        def get_doctors_by_specialist(specialist: str):
            """Return doctors for a given specialization."""
            from sqlalchemy.exc import SQLAlchemyError

            try:
                with Session(engine) as session:
                    if specialist:
                        stmt = select(Doctor).where(Doctor.specialist == specialist)
                    else:
                        stmt = select(Doctor)
                    doctors = session.exec(stmt).all()
                # Return JSON-serializable list
                return [
                    {
                        "id": d.id,
                        "name": d.name,
                        "specialist": d.specialist,
                        "hospital": d.hospital,
                        "email": d.email,
                        "contact_no": d.contact_no,
                        "education": d.education,
                    }
                    for d in doctors
                ]
            except SQLAlchemyError as e:
                logger.error("DB Error:", e)
                return []


        specialist_agent = Agent(
            name="Specialist Finder",
            instructions="""
            You are an agent that helps find medical specialists based on the provided medical summary.
            Given a medical summary, identify the relevant specialist type (e.g., cardiologist, neurologist).
            Use the `get_doctors_by_specialist` tool to retrieve a list of available doctors in that specialization.
            Return a brief confirmation message including the specialist type, sample doctor's name, and contact information,hospital, and education.

            Return the final information as plain text (no JSON).
            """,
            tools=[get_doctors_by_specialist],
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.3),
        )
        specialist = await Runner.run(specialist_agent, medical_summary)
        specialist_info = getattr(specialist, "final_output", None) or getattr(specialist, "output", None)

        # Email sending
        EMAIL_AGENT_INSTRUCTIONS = """You are able to send a nicely formatted HTML email based on a detailed report.
        You will be provided with a detailed report. You should use your tool to send one email, providing the 
        report converted into clean, well presented HTML with an appropriate subject line.
        
        Schedule appointment email with the provided medical summary and specialist information.
        
        Write the exact Medical Summary what is provided to you.
        
        Example Schedule Appointment: (In Tabular Format)
        
        +----------------+----------------+----------------+----------------+----------------+----------------+-----------------------------------+
        | Doctor Name    | Specialist     | Hospital       | Contact No     | Email          | Education               | Schedule Date            |
        +----------------+----------------+----------------+----------------+----------------+----------------+-----------------------------------+
        | Dr. John Doe   | Cardiologist   | City Hospital  | 123-456-7890   | dr.johndoe@hospital.com | MD, Cardiology | 2023-10-15 10:00 AM (EST)|
        +----------------+----------------+----------------+----------------+----------------+----------------+-----------------------------------+

        If possible, attach the original PDF file to the email for reference.
        """
        

        @function_tool
        def send_email(subject: str, html_body: str, attachment_path: Optional[str] = None) -> Dict[str, str]:
            """Send an email using SendGrid with the given subject and HTML body. Optionally attach a file."""
            SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")

            if not SENDGRID_API_KEY:
                return {"status": "error", "message": "SENDGRID_API_KEY not found"}

            try:
                message = Mail(
                    from_email=os.getenv("FROM_EMAIL", "justdevaplaying@gmail.com"),
                    to_emails=os.getenv("TO_EMAIL", email),
                    subject=subject,
                    html_content=html_body,
                )

                # Attach file if provided
                if attachment_path and os.path.exists(attachment_path):
                    with open(attachment_path, 'rb') as f:
                        file_data = f.read()
                    encoded_file = base64.b64encode(file_data).decode()
                    filename = os.path.basename(attachment_path)
                    mimetype, _ = mimetypes.guess_type(attachment_path)
                    if not mimetype:
                        mimetype = 'application/octet-stream'  # Fallback

                    attachment = Attachment(
                        FileContent(encoded_file),
                        FileName(filename),
                        FileType(mimetype),
                        Disposition('attachment')
                    )
                    message.add_attachment(attachment)

                sendgrid_client = SendGridAPIClient(SENDGRID_API_KEY)
                response = sendgrid_client.send(message)

                return {
                    "status": "success",
                    "code": response.status_code,
                }

            except Exception as e:
                return {"status": "error", "message": str(e)}
        
        email_agent = Agent(
            name="Email agent",
            instructions=EMAIL_AGENT_INSTRUCTIONS,
            tools=[send_email],
            model="gpt-4o-mini",
        )
        # Pass file_path in the input so the agent can use it
        input_text = f"Send the following medical appointment details to {email}:\n\n Specialist Information: {specialist_info} \n\n Medical Summary: {medical_summary}"
        if file_path:
            input_text += f"\n\nAttach the PDF from: {file_path}"
        final_email_content = await Runner.run(email_agent, input_text)
        final_email_content_output = getattr(final_email_content, "final_output", None) or getattr(final_email_content, "output", None)
        return str(final_email_content_output)


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

        # Pass file_path to schedule_appointment
        result = await health_manager.run(file_path, email)

    except Exception as e:
        logger.error(f"Health analysis error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error")
    finally:
        # Clean up the temporary file after everything is done
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except Exception:
                logger.warning("Failed to delete temporary file: %s", file_path)

    return {
        "status": "success",
        "summary_email_content": result,
    }