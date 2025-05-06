"""
Document Summarizer API - Data Models and Utilities with Async Support
--------------------------------------------------------------------
This file contains all Pydantic models and async utility functions for the API.
"""
import os
import uuid
import logging
import json
import time
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from fastapi import HTTPException
from pydantic import BaseModel, Field

# Import the AsyncEnhancedSummarizer class and AsyncDocumentProcessor
from enhanced_summarizer import AsyncEnhancedSummarizer, SMALL_DOC_MODEL, LARGE_DOC_MODEL
from document_processor import AsyncDocumentProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', './uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'md', 'html'}

# Define models for request/response
class SummaryRequest(BaseModel):
    model: str = Field(..., description="OpenRouter model ID (e.g., 'anthropic/claude-3-opus')")
    temperature: float = Field(0.3, description="Temperature parameter (0.0-1.0)")
    max_tokens: int = Field(1000, description="Maximum tokens in the summary")
    chunk_size: int = Field(4000, description="Size of text chunks for processing")
    chunk_overlap: int = Field(200, description="Overlap between chunks")
    chain_type: str = Field("refine", description="Summarization chain type ('map_reduce' or 'refine')")
    custom_prompt: Optional[str] = Field(None, description="Custom prompt template (include {text} placeholder)")

class SummaryResponse(BaseModel):
    summary: str = Field(..., description="Generated summary")
    metadata: Dict[str, Any] = Field(..., description="Metadata about the summarization process")

class SummaryTask(BaseModel):
    task_id: str = Field(..., description="Unique task ID")
    status: str = Field(..., description="Task status (pending, processing, completed, failed)")
    created_at: float = Field(..., description="Task creation timestamp")
    completed_at: Optional[float] = Field(None, description="Task completion timestamp")
    result: Optional[SummaryResponse] = Field(None, description="Summary result (if completed)")
    error: Optional[str] = Field(None, description="Error message (if failed)")
    document_text: Optional[str] = None  # Document text for QA (not included in API responses)

class QARequest(BaseModel):
    task_id: str = Field(..., description="Task ID of the processed document")
    question: str = Field(..., description="Question to answer")
    model: Optional[str] = Field(None, description="Model to use for answering (default: auto-select)")

class QATask(BaseModel):
    task_id: str = Field(..., description="QA task ID")
    document_task_id: str = Field(..., description="Document task ID")
    question: str = Field(..., description="Question asked")
    status: str = Field(..., description="Task status (pending, processing, completed, failed)")
    created_at: float = Field(..., description="Task creation timestamp")
    completed_at: Optional[float] = Field(None, description="Task completion timestamp")
    answer: Optional[str] = Field(None, description="Answer to the question")
    error: Optional[str] = Field(None, description="Error message (if failed)")
    model: str = Field(..., description="Model used for answering")
    suggested_questions: Optional[List[str]] = Field(None, description="Suggested follow-up questions")

class QAResponse(BaseModel):
    task_id: str = Field(..., description="QA task ID")
    document_task_id: str = Field(..., description="Document task ID")
    question: str = Field(..., description="Question asked")
    status: str = Field(..., description="Task status")
    created_at: float = Field(..., description="Task creation timestamp")
    completed_at: Optional[float] = Field(None, description="Task completion timestamp")
    answer: Optional[str] = Field(None, description="Answer to the question")
    error: Optional[str] = Field(None, description="Error message (if failed)")
    model: str = Field(..., description="Model used for answering")
    suggested_questions: Optional[List[str]] = Field(None, description="Suggested follow-up questions")

class SuggestedQuestions(BaseModel):
    questions: List[str] = Field(..., description="List of suggested questions")
    source: str = Field("summary", description="Source of suggested questions (summary or document)")

# In-memory storage for tasks
tasks = {}
qa_tasks = {}

def allowed_file(filename: str) -> bool:
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

async def process_document_task(
    task_id: str,
    file_path: str,
    model: str,
    temperature: float,
    max_tokens: int,
    chunk_size: int,
    chunk_overlap: int,
    chain_type: str,
    summary_length: str,
    focus_areas: List[str]
):
    """Async background task to process document and generate summary"""
    try:
        # Update task status to processing
        tasks[task_id].status = "processing"
        
        # Initialize document processor
        doc_processor = AsyncDocumentProcessor(verbose=True)
        
        # Process document to extract text asynchronously
        text, file_metadata = await doc_processor.process_document(file_path)
        
        # Store the document text for later QA use
        tasks[task_id].document_text = text
        
        # Log the model being used for this specific task
        logger.info(f"Task {task_id}: Using model {model}")
        
        # Build custom prompt based on focus areas
        focus_map = {
            "key_points": "key points and main arguments",
            "methodology": "methodology and procedures",
            "examples": "examples and supporting evidence",
            "conclusions": "conclusions and findings",
            "implications": "implications and applications",
            "technical": "technical details and specifications"
        }
        
        focus_instructions = [focus_map[area] for area in focus_areas if area in focus_map]
        
        if focus_instructions:
            focus_text = ", ".join(focus_instructions[:-1])
            if len(focus_instructions) > 1:
                focus_text += f", and {focus_instructions[-1]}"
            else:
                focus_text = focus_instructions[0]
        else:
            focus_text = "all important aspects"
            
        custom_prompt = f"""
        You are a professional summarizer with expertise in creating clear, concise, and comprehensive summaries.

        Please summarize the following text:

        {{text}}

        Your summary should be {summary_length.lower()} in length and focus primarily on {focus_text}.
        Structure your summary with a logical flow. Include an introduction, body, and conclusion if appropriate for the length.
        """
        
        # Initialize summarizer with the specified model
        summarizer = AsyncEnhancedSummarizer(
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chain_type=chain_type,
            http_referer="https://document-summarizer-api.example.com/"
        )
        
        # Log the model that was used
        logger.info(f"Task {task_id}: Initialized summarizer with model {model}")
        
        # Generate summary asynchronously
        result = await summarizer.summarize(
            text,
            is_file_path=False,
            custom_prompt=custom_prompt,
            verbose=True
        )
        
        # Verify that the model in the result matches what was requested
        logger.info(f"Task {task_id}: Summary generated using model {result['metadata']['model']}")
        
        # Add QA availability to metadata
        result["metadata"]["qa_available"] = True
        
        # Update task status
        tasks[task_id].status = "completed"
        tasks[task_id].completed_at = time.time()
        tasks[task_id].result = SummaryResponse(
            summary=result["summary"],
            metadata=result["metadata"]
        )
        
        # Clean up
        try:
            os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Error removing temporary file {file_path}: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        tasks[task_id].status = "failed"
        tasks[task_id].error = str(e)
        tasks[task_id].completed_at = time.time()
        
        # Clean up on error
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as cleanup_error:
            logger.warning(f"Error during cleanup: {str(cleanup_error)}")

async def process_qa_task(
    qa_task_id: str,
    document_task_id: str,
    question: str,
    model: Optional[str]
):
    """Process a question-answering task in the background asynchronously"""
    try:
        # Update QA task status to processing
        qa_tasks[qa_task_id].status = "processing"
        
        # Get document text from the original document task
        if document_task_id not in tasks:
            raise ValueError(f"Document task {document_task_id} not found")
            
        document_task = tasks[document_task_id]
        
        if not hasattr(document_task, 'document_text') or document_task.document_text is None:
            raise ValueError("Document text not available for this task")
            
        document_text = document_task.document_text
        document_length = len(document_text)
        
        # Auto-select appropriate model if not provided
        if model is None:
            model = AsyncEnhancedSummarizer.get_model_for_document_size(document_length)
            logger.info(f"Auto-selected model {model} for QA based on document length ({document_length} chars)")
            qa_tasks[qa_task_id].model = model
            
        # Create an AsyncEnhancedSummarizer instance for QA
        qa_model = AsyncEnhancedSummarizer(
            model_name=model,
            temperature=0.2,  # Lower temperature for factual responses
            max_tokens=1000,  # Appropriate limit for answers
            http_referer="https://document-summarizer-api.example.com/"
        )
        
        # Get the answer asynchronously
        answer, metadata = await qa_model.answer_question(document_text, question, model)
        
        # Generate suggested follow-up questions asynchronously
        summary = document_task.result.summary if document_task.result else None
        suggested_questions = await generate_suggested_questions(
            document_text=document_text,
            summary=summary,
            previous_question=question,
            previous_answer=answer,
            count=3  # Fewer follow-up questions
        )
        
        # Update the QA task
        qa_tasks[qa_task_id].status = "completed"
        qa_tasks[qa_task_id].completed_at = time.time()
        qa_tasks[qa_task_id].answer = answer
        # Add the suggested questions to the response
        qa_tasks[qa_task_id].suggested_questions = suggested_questions
        
        logger.info(f"QA task {qa_task_id} completed with model {model}")
        
    except Exception as e:
        logger.error(f"Error processing QA task: {str(e)}")
        qa_tasks[qa_task_id].status = "failed"
        qa_tasks[qa_task_id].error = str(e)
        qa_tasks[qa_task_id].completed_at = time.time()

async def generate_suggested_questions(
    document_text: str,
    summary: Optional[str] = None,
    previous_question: Optional[str] = None,
    previous_answer: Optional[str] = None,
    count: int = 5
) -> List[str]:
    """Generate suggested questions based on document content and optionally previous Q&A asynchronously"""
    # Choose an appropriate model - using a smaller model for speed
    model = SMALL_DOC_MODEL
    
    # Create an AsyncEnhancedSummarizer instance for generating questions
    question_generator = AsyncEnhancedSummarizer(
        model_name=model,
        temperature=0.7,  # Higher temperature for creative questions
        max_tokens=500,
        http_referer="https://document-summarizer-api.example.com/"
    )
    
    # Build the prompt based on whether we have previous Q&A or just the summary
    if previous_question and previous_answer:
        prompt = f"""
        Based on the document and previous Q&A, suggest {count} follow-up questions that would be interesting to ask.
        
        Document summary:
        {summary if summary else "No summary available"}
        
        Previous question: {previous_question}
        Previous answer: {previous_answer}
        
        Generate {count} different follow-up questions that explore new aspects related to the previous Q&A or other important topics in the document.
        Format your response as a numbered list of questions only, without any explanations or other text.
        """
    else:
        prompt = f"""
        Based on the following document, generate {count} interesting questions that someone might want to ask about the content.
        
        Document summary:
        {summary if summary else "No summary available"}
        
        Focus on questions that explore the main topics, key findings, important details, and implications in the document.
        Include a mix of factual and analytical questions.
        Format your response as a numbered list of questions only, without any explanations or other text.
        """
    
    # Generate questions asynchronously
    logger.info(f"Generating suggested questions based on {'previous Q&A' if previous_question else 'document'}")
    response, _ = await question_generator.answer_question(document_text, prompt)
    
    # Parse the response to extract questions
    questions = []
    for line in response.strip().split('\n'):
        line = line.strip()
        # Remove numbering and any other formatting
        if line and (line[0].isdigit() or line[0] in ['•', '-', '*']):
            # Remove the numbering/bullet and any trailing punctuation
            clean_line = line.split('.', 1)[-1].strip() if '.' in line[:3] else line
            clean_line = clean_line.lstrip('-•*').strip()
            clean_line = clean_line.rstrip('?') + '?'  # Ensure it ends with a question mark
            if clean_line and len(clean_line) > 10:  # Minimum question length
                questions.append(clean_line)
    
    # If parsing failed, create some generic questions
    if len(questions) < 2:
        logger.warning("Failed to parse questions from model response, using fallback questions")
        questions = [
            "What are the main points of this document?",
            "Can you explain the key findings in more detail?",
            "What are the implications of this information?",
            "How does this relate to the current state of the field?",
            "What methodologies were used in this document?"
        ]
    
    # Limit to requested count
    return questions[:count]