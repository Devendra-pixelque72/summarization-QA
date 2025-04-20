"""
Document Summarizer Microservice
--------------------------------
FastAPI microservice for document summarization and QA using OpenRouter LLMs.
"""
import os
import base64
import json
import logging
import uuid
import time
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from dotenv import load_dotenv

# Import models and processing functions
from models import (
    SummaryTask, QATask, QARequest, QAResponse, SuggestedQuestions,
    tasks, qa_tasks, allowed_file, process_document_task, process_qa_task, 
    generate_suggested_questions, UPLOAD_FOLDER
)

# Import from enhanced_summarizer for model fetching
from enhanced_summarizer import EnhancedSummarizer, SMALL_DOC_MODEL, LARGE_DOC_MODEL

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for OpenRouter API key
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY not found in environment variables. Service will not work properly.")

# Initialize FastAPI app
app = FastAPI(
    title="Document Summarizer & QA API",
    description="API for summarizing documents and answering questions using OpenRouter LLMs",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Serve the index.html file directly
@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the index.html file."""
    try:
        with open("index.html", "r") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        logger.error("index.html file not found")
        return HTMLResponse(content="<html><body><h1>Error: index.html file not found</h1></body></html>")
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        return HTMLResponse(content=f"<html><body><h1>Error: {str(e)}</h1></body></html>")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }

@app.get("/models")
async def get_models():
    """Retrieve available models from OpenRouter"""
    try:
        if not OPENROUTER_API_KEY:
            raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")
            
        models = EnhancedSummarizer.get_openrouter_models()
        
        # Highlight the recommended models for document summarization and QA
        for model in models:
            if model["name"] == SMALL_DOC_MODEL:
                model["description"] = f"[RECOMMENDED FOR SMALL DOCS] {model.get('description', '')}"
            elif model["name"] == LARGE_DOC_MODEL:
                model["description"] = f"[RECOMMENDED FOR LARGE DOCS] {model.get('description', '')}"
                
        return {"models": models}
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

@app.post("/summarize", response_model=SummaryTask)
async def summarize_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form(...),
    temperature: float = Form(0.3),
    max_tokens: int = Form(1000),
    summary_length: str = Form("Moderate"),
    focus_areas: str = Form('["key_points", "conclusions"]'),
    chunk_size: int = Form(4000),
    chunk_overlap: int = Form(200),
    chain_type: str = Form("refine")
):
    """
    Upload a document and generate a summary using the specified model.
    This endpoint starts an asynchronous process and returns a task ID for checking the status.
    """
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")
    
    # Log the incoming request parameters
    logger.info(f"Summarize request received: model={model}, length={summary_length}")
    
    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
        
    file_extension = os.path.splitext(file.filename)[1].lower()
    valid_extensions = ['.pdf', '.docx', '.txt', '.md', '.html']
    
    if file_extension not in valid_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported formats: {', '.join(valid_extensions)}"
        )
    
    # Validate chain type
    if chain_type not in ["refine", "map_reduce"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid chain_type. Use 'refine' or 'map_reduce'."
        )
    
    # Parse focus areas from JSON string
    try:
        focus_areas_list = json.loads(focus_areas)
        if not isinstance(focus_areas_list, list):
            focus_areas_list = ["key_points", "conclusions"]
        
        logger.info(f"Focus areas: {focus_areas_list}")
    except Exception as e:
        logger.warning(f"Error parsing focus areas: {str(e)}")
        focus_areas_list = ["key_points", "conclusions"]
    
    # Ensure at least one focus area is selected
    if not focus_areas_list:
        focus_areas_list = ["key_points"]
        logger.warning("No focus areas provided, using default: key_points")
    
    # Create a temporary file
    try:
        # Create a unique filename
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        tmp_file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # Write uploaded file content to the temp file
        content = await file.read()
        with open(tmp_file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")
    
    # Create a task ID
    task_id = base64.urlsafe_b64encode(os.urandom(16)).decode('ascii')
    
    # Initialize task
    tasks[task_id] = SummaryTask(
        task_id=task_id,
        status="pending",
        created_at=time.time(),
        completed_at=None,
        result=None,
        error=None
    )
    
    # Log the task creation
    logger.info(f"Created task {task_id} for model {model}")
    
    # Add task to background tasks
    background_tasks.add_task(
        process_document_task,
        task_id,
        tmp_file_path,
        model,
        temperature,
        max_tokens,
        chunk_size,
        chunk_overlap,
        chain_type,
        summary_length,
        focus_areas_list
    )
    
    return tasks[task_id]

@app.get("/tasks/{task_id}", response_model=SummaryTask)
async def get_task_status(task_id: str):
    """Get the status of a summarization task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Return the task without the document_text field (to avoid large responses)
    task_response = tasks[task_id].dict(exclude={"document_text"})
    return task_response

@app.post("/qa", response_model=QATask)
async def ask_question(
    background_tasks: BackgroundTasks,
    qa_request: QARequest
):
    """
    Ask a question about a previously processed document.
    
    Args:
        qa_request: QA request containing task_id, question, and optional model
    """
    # Validate document task
    if qa_request.task_id not in tasks:
        raise HTTPException(status_code=404, detail="Document not found. Please upload a document first.")
    
    # Get the document task
    document_task = tasks[qa_request.task_id]
    
    # Check if document processing is complete
    if document_task.status != "completed":
        raise HTTPException(status_code=400, detail="Document processing hasn't completed yet.")
    
    # Check if document text is available
    if not hasattr(document_task, 'document_text') or document_task.document_text is None:
        raise HTTPException(status_code=400, detail="Document text not available for Q&A.")
    
    # Create QA task ID
    qa_task_id = f"qa_{uuid.uuid4().hex[:8]}"
    
    # Get document length for auto model selection if needed
    document_length = len(document_task.document_text)
    
    # Select model if not provided
    model = qa_request.model
    if not model:
        model = EnhancedSummarizer.get_model_for_document_size(document_length)
        logger.info(f"Auto-selected model {model} for QA based on document length ({document_length} chars)")
    
    # Create QA task
    qa_tasks[qa_task_id] = QATask(
        task_id=qa_task_id,
        document_task_id=qa_request.task_id,
        question=qa_request.question,
        status="pending",
        created_at=time.time(),
        completed_at=None,
        answer=None,
        error=None,
        model=model
    )
    
    # Process QA in background
    background_tasks.add_task(
        process_qa_task,
        qa_task_id,
        qa_request.task_id,
        qa_request.question,
        model
    )
    
    return qa_tasks[qa_task_id]

@app.get("/qa/{qa_task_id}", response_model=QAResponse)
async def get_qa_status(qa_task_id: str):
    """Get the status of a QA task with suggested follow-up questions"""
    if qa_task_id not in qa_tasks:
        raise HTTPException(status_code=404, detail="QA task not found")
    
    # Return the task with suggested questions if available
    task = qa_tasks[qa_task_id]
    
    # Create a QAResponse that includes suggested questions
    response = QAResponse(
        task_id=task.task_id,
        document_task_id=task.document_task_id,
        question=task.question,
        status=task.status,
        created_at=task.created_at,
        completed_at=task.completed_at,
        answer=task.answer,
        error=task.error,
        model=task.model,
        suggested_questions=getattr(task, 'suggested_questions', None)
    )
    
    return response

@app.get("/suggest-questions/{task_id}", response_model=SuggestedQuestions)
async def get_suggested_questions(
    task_id: str,
    previous_question: Optional[str] = None,
    previous_answer: Optional[str] = None,
    count: int = 5
):
    """Get suggested questions for a document"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Document not found")
        
    task = tasks[task_id]
    
    if task.status != "completed":
        raise HTTPException(status_code=400, detail="Document processing not completed")
        
    if not hasattr(task, 'document_text') or task.document_text is None:
        raise HTTPException(status_code=400, detail="Document text not available")
    
    # Get the document text and summary
    document_text = task.document_text
    summary = task.result.summary if task.result else None
    
    # Generate suggested questions
    questions = await generate_suggested_questions(
        document_text=document_text,
        summary=summary,
        previous_question=previous_question,
        previous_answer=previous_answer,
        count=count
    )
    
    return SuggestedQuestions(questions=questions, source="summary" if summary else "document")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)