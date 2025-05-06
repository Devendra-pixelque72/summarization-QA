"""
Enhanced document summarization and QA using OpenRouter API directly with async support.
"""
import os
import time
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import logging
import aiohttp

# Import the direct OpenRouter client
from openrouter_patch import ChatOpenRouter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define model configurations for different document sizes
SMALL_DOC_MODEL = "openai/gpt-3.5-turbo"
LARGE_DOC_MODEL = "google/gemini-flash-1.5"

class AsyncEnhancedSummarizer:
    """Enhanced document summarization and QA using OpenRouter API directly with async support."""
    
    # Define the default prompt as a class variable
    DEFAULT_PROMPT = """
You are a professional summarizer with expertise in creating clear, concise, and comprehensive summaries.

Please summarize the following text:

{text}

Focus on the main points, key arguments, and important details while maintaining the original meaning.
Structure your summary with a logical flow. Include an introduction, body, and conclusion.
"""
    
    # Define default QA prompt
    DEFAULT_QA_PROMPT = """
You are an AI assistant answering questions about documents.

Document text:
{text}

Based only on the information in the document above, please answer the following question:
{question}

If the answer cannot be found in the document, say "I don't have enough information in the document to answer this question."

Answer:
"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        provider: str = "openrouter",
        chain_type: str = "refine",
        http_referer: str = "https://document-summarizer.example.com"
    ):
        """
        Initialize the summarizer with customizable parameters.
        
        Args:
            model_name: Name of the model to use (provider-specific)
            temperature: Temperature parameter for generation (0.0-1.0)
            max_tokens: Maximum tokens in the response
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            provider: Provider to use (openrouter is the only supported option)
            chain_type: Summarization chain type (refine or map_reduce)
            http_referer: HTTP referer header for OpenRouter API calls
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.http_referer = http_referer
        self.provider = provider
        self.chain_type = chain_type
        
        # Get API key from environment
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key not found in environment variables (OPENROUTER_API_KEY)")
        
        # Set default model if not specified
        if model_name is None:
            model_name = SMALL_DOC_MODEL
            
        self.model_name = model_name
        logger.info(f"Initializing summarizer with model: {model_name}")
        
        # Initialize the OpenRouter client
        self.client = ChatOpenRouter(
            openrouter_api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            http_referer=http_referer,
            provider=provider,
            chain_type=chain_type
        )
    
    @staticmethod
    async def get_openrouter_models() -> List[Dict[str, Any]]:
        """Fetch available models from OpenRouter API asynchronously"""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            logger.error("OpenRouter API key not found in environment variables")
            return []
            
        try:
            models = await ChatOpenRouter.get_available_models()
            logger.info(f"Retrieved {len(models)} models from OpenRouter")
            return models
        except Exception as e:
            logger.error(f"Error fetching OpenRouter models: {str(e)}")
            return []
    
    @staticmethod
    def get_model_for_document_size(text_length: int) -> str:
        """
        Select an appropriate model based on document length.
        
        Args:
            text_length: Length of the document text in characters
            
        Returns:
            Model name appropriate for the document size
        """
        # Threshold for large documents (approximately 50 pages of text)
        # Based on an estimate of 2000 characters per page
        LARGE_DOC_THRESHOLD = 100000  # 100K characters
        
        if text_length > LARGE_DOC_THRESHOLD:
            logger.info(f"Document length {text_length} chars exceeds threshold, using large document model")
            return LARGE_DOC_MODEL
        else:
            logger.info(f"Document length {text_length} chars below threshold, using standard model")
            return SMALL_DOC_MODEL
    
    async def read_document(self, file_path: str) -> str:
        """Read document from file asynchronously."""
        try:
            # Use aiofiles to read the file asynchronously
            try:
                import aiofiles
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
                    text = await file.read()
                return text
            except ImportError:
                # Fall back to synchronous read if aiofiles is not available
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                return text
        except UnicodeDecodeError:
            # Try with a different encoding if UTF-8 fails
            try:
                import aiofiles
                async with aiofiles.open(file_path, 'r', encoding='latin-1') as file:
                    text = await file.read()
                return text
            except ImportError:
                # Fall back to synchronous read
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
                return text
        except Exception as e:
            raise ValueError(f"Error reading file '{file_path}': {e}")
    
    def chunk_document(self, text: str) -> List[str]:
        """Split document into chunks."""
        # Simple chunking by paragraphs then by size
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, add current chunk to list and start a new one
            if len(current_chunk) + len(paragraph) + 2 > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)
            
        # Handle overlap if needed
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapping_chunks = []
            for i in range(len(chunks)):
                if i == 0:
                    overlapping_chunks.append(chunks[i])
                else:
                    # Get the end of the previous chunk to create overlap
                    prev_chunk = chunks[i-1]
                    overlap_size = min(self.chunk_overlap, len(prev_chunk))
                    overlap_text = prev_chunk[-overlap_size:] if overlap_size > 0 else ""
                    overlapping_chunks.append(overlap_text + chunks[i])
            return overlapping_chunks
            
        return chunks
    
    async def summarize_chunk(self, chunk: str, custom_prompt: Optional[str] = None) -> str:
        """Summarize a single chunk of text asynchronously."""
        prompt_template = custom_prompt if custom_prompt else self.DEFAULT_PROMPT
        prompt = prompt_template.replace("{text}", chunk)
        
        # Log which model is being used for this chunk
        logger.info(f"Summarizing chunk with model: {self.model_name}")
        
        summary, _ = await self.client.summarize(chunk, custom_prompt=prompt_template)
        return summary
    
    async def answer_question(
        self,
        document_text: str,
        question: str,
        model_name: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Answer a question based on the document text asynchronously.
        
        Args:
            document_text: The document text to reference
            question: The question to answer
            model_name: Optional model name to use (defaults to self.model_name)
            
        Returns:
            Tuple of (answer, metadata)
        """
        # Use provided model or default to class model
        model_to_use = model_name or self.model_name
        
        # If document is very large, consider using a model with larger context window
        if len(document_text) > 100000 and not model_name:  # ~100K chars
            model_to_use = LARGE_DOC_MODEL
            logger.info(f"Using large document model {model_to_use} for QA due to document size")
        
        # Create QA client with appropriate model
        qa_client = None
        if model_to_use != self.model_name:
            # Create a new client with the selected model
            qa_client = ChatOpenRouter(
                openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
                model_name=model_to_use,
                temperature=0.2,  # Lower temperature for factual answers
                max_tokens=500,  # Shorter responses for QA
                http_referer=self.http_referer
            )
            logger.info(f"Created new QA client with model: {model_to_use}")
        else:
            # Use existing client
            qa_client = self.client
            logger.info(f"Using existing client with model: {model_to_use} for QA")
        
        # Format the QA prompt
        qa_prompt = self.DEFAULT_QA_PROMPT.replace("{text}", document_text).replace("{question}", question)
        
        # Get the answer
        try:
            if qa_client == self.client:
                # If using the same client, use its summarize method
                answer, metadata = await qa_client.summarize("", custom_prompt=qa_prompt)
            else:
                # If using a new client, use its answer_question method
                answer, metadata = await qa_client.answer_question(document_text, question)
            
            return answer, metadata
        except Exception as e:
            logger.error(f"Error in QA: {str(e)}")
            return f"Error generating answer: {str(e)}", {"error": str(e)}
    
    async def summarize_chunks_parallel(self, chunks: List[str], custom_prompt: Optional[str] = None) -> List[str]:
        """Summarize multiple chunks in parallel asynchronously."""
        if not chunks:
            return []
            
        # Create tasks for all chunks
        tasks = [self.summarize_chunk(chunk, custom_prompt) for chunk in chunks]
        
        # Execute all summarization tasks in parallel
        chunk_summaries = await asyncio.gather(*tasks)
        return chunk_summaries
    
    async def summarize(
        self, 
        input_source: str, 
        is_file_path: bool = True,
        custom_prompt: Optional[str] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Summarize a document with detailed metrics asynchronously.
        
        Args:
            input_source: Path to document file OR text content
            is_file_path: If True, input_source is a file path; if False, it's text content
            custom_prompt: Optional custom prompt template
            verbose: If True, print detailed information during processing
            
        Returns:
            Dictionary containing summary and metadata
        """
        start_time = time.time()
        
        if verbose and is_file_path:
            logger.info(f"Reading document: {input_source}")
        
        # Get text content
        if is_file_path:
            text = await self.read_document(input_source)
            doc_path = input_source
        else:
            text = input_source
            doc_path = "text_input"
            
        doc_length = len(text)
        
        if verbose:
            logger.info(f"Document length: {doc_length} characters")
            logger.info(f"Chunking document (size: {self.chunk_size}, overlap: {self.chunk_overlap})")
        
        # Chunk document
        chunks = self.chunk_document(text)
        num_chunks = len(chunks)
        
        if verbose:
            logger.info(f"Document split into {num_chunks} chunks")
            logger.info(f"Using model: {self.model_name}")
        
        # Process based on number of chunks
        if num_chunks == 0:
            logger.warning("No text chunks to process")
            summary = "No content to summarize."
            metadata = {}
        elif num_chunks == 1:
            # If only one chunk, summarize it directly
            if verbose:
                logger.info("Single chunk, generating summary directly")
            summary, metadata = await self.client.summarize(chunks[0], custom_prompt=custom_prompt or self.DEFAULT_PROMPT)
        else:
            # For multiple chunks, summarize each chunk in parallel and then combine
            if verbose:
                logger.info(f"Processing {num_chunks} chunks in parallel")
                
            # First pass: summarize each chunk in parallel
            chunk_summaries = await self.summarize_chunks_parallel(chunks, custom_prompt)
                
            # Second pass: combine the summaries
            if verbose:
                logger.info("Combining chunk summaries")
                
            combined_text = "\n\n".join(chunk_summaries)
            
            # Create a final summary prompt
            final_prompt = """
You are a professional summarizer with expertise in creating clear, concise, and comprehensive summaries.

The following text is a collection of summaries from different sections of a document.
Please create a cohesive final summary that integrates all the information:

{text}

Provide a well-structured summary with a logical flow. Include an introduction, body, and conclusion.
"""
            
            summary, metadata = await self.client.summarize(combined_text, custom_prompt=final_prompt)
        
        # Estimate token usage
        # Rough estimate: 1 token ≈ 4 characters
        estimated_input_tokens = len(text) // 4
        estimated_output_tokens = len(summary) // 4
        
        token_usage = {
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_total_tokens": estimated_input_tokens + estimated_output_tokens
        }
        
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)
        
        # Prepare response
        response = {
            "summary": summary,
            "metadata": {
                "document_path": doc_path,
                "document_length": doc_length,
                "compression_ratio": round(len(summary) / doc_length, 3) if doc_length > 0 else 0,
                "num_chunks": num_chunks,
                "model": self.model_name,
                "provider": self.provider,
                "chain_type": self.chain_type,
                "token_usage": token_usage,
                "processing_time_seconds": processing_time
            }
        }
        
        if verbose:
            logger.info(f"Summarization complete - {processing_time} seconds")
            logger.info(f"Compression ratio: {response['metadata']['compression_ratio']}")
            logger.info(f"Estimated token usage: {token_usage['estimated_total_tokens']} tokens")
        
        return response