import os
import json
import logging
import tempfile
from typing import Optional, List

import requests
from fastapi import APIRouter, UploadFile, File
from PyPDF2 import PdfReader


router = APIRouter()

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


@router.post("/test-extract-text")
async def test_extract_text(file: UploadFile = File(...)):
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_file_path = temp_file.name
    
    try:
        text = extract_text_from_file(temp_file_path)
        return {"extracted_text": text}
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)