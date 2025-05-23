"""
OpenRouter integration with LangChain using the latest API patterns.
"""
from typing import Any, Dict, List, Optional, Sequence, Union, Mapping, Type, cast, Tuple
import json
import logging
import requests
import os
from urllib.parse import urlparse

from langchain_core.callbacks.manager import CallbackManagerForLLMRun, Callbacks
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.pydantic_v1 import Field, root_validator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatOpenRouter(BaseChatModel):
    """Chat model that uses OpenRouter API."""
    
    openrouter_api_key: str
    model_name: str = "openai/gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    provider: str = "openrouter"
    chain_type: str = "refine"
    
    # HTTP config
    openrouter_url: str = "https://openrouter.ai/api/v1/chat/completions"
    timeout: Optional[int] = 120
    http_referer: str = "https://your-app-domain.com/"
    x_title: Optional[str] = "Document Summarizer API"
    
    # Debug options
    debug_mode: bool = True
    
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists."""
        if not values.get("openrouter_api_key"):
            raise ValueError(
                "OpenRouter API key must be provided. "
                "Get one at https://openrouter.ai"
            )
        return values
    
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenRouter API."""
        params = {
            "model": self.model_name,
            "temperature": self.temperature,
        }
        
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
            
        if self.top_p is not None:
            params["top_p"] = self.top_p
            
        return params
    
    def _convert_message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        """Convert a LangChain message to a dictionary for the OpenRouter API."""
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            return {"role": "assistant", "content": message.content}
        elif isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        elif isinstance(message, ChatMessage):
            return {"role": message.role, "content": message.content}
        else:
            raise ValueError(f"Got unknown message type: {message}")
    
    def _create_chat_result(self, response: Dict[str, Any]) -> ChatResult:
        """Process the API response and create a ChatResult."""
        generations = []
        
        # Safely extract choices
        choices = response.get("choices", [])
        
        if not choices:
            logger.warning("No choices in response, creating fallback response")
            fallback_msg = "The model did not return any response. Please try again or try a different model."
            
            generations.append(
                ChatGeneration(
                    message=AIMessage(content=fallback_msg),
                    generation_info={"fallback_response": True}
                )
            )
        else:
            for choice in choices:
                message = choice.get("message", {})
                message_content = message.get("content", "")
                
                if not message_content:
                    logger.warning(f"Empty content in choice: {choice}")
                    message_content = "[No content returned by model]"
                
                generation_info = {
                    "finish_reason": choice.get("finish_reason"),
                    "index": choice.get("index"),
                }
                
                generations.append(
                    ChatGeneration(
                        message=AIMessage(content=message_content),
                        generation_info=generation_info,
                    )
                )
        
        token_usage = response.get("usage", {})
        llm_output = {"token_usage": token_usage, "model_name": self.model_name}
        
        return ChatResult(generations=generations, llm_output=llm_output)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion using OpenRouter API."""
        # Ensure HTTP-Referer has a valid format
        referer = self.http_referer
        if referer and not (referer.startswith('http://') or referer.startswith('https://')):
            referer = f"https://{referer}"
        
        # For testing, provide a default referer if none exists
        if not referer:
            referer = "https://document-summarizer-api.example.com/"
            
        # Validate referer
        parsed_referer = urlparse(referer)
        if not all([parsed_referer.scheme, parsed_referer.netloc]):
            logger.warning(f"Invalid referer URL: {referer}, using default")
            referer = "https://document-summarizer-api.example.com/"
            
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "HTTP-Referer": referer,
            "X-Title": self.x_title or "Document Summarizer API",
            "Content-Type": "application/json",
        }
        
        # Log headers for debugging (except API key)
        if self.debug_mode:
            debug_headers = headers.copy()
            debug_headers["Authorization"] = "Bearer [REDACTED]"
            logger.info(f"Request headers: {debug_headers}")
            logger.info(f"Request URL: {self.openrouter_url}")
        
        # Format messages for the API
        formatted_messages = [self._convert_message_to_dict(m) for m in messages]
        
        # Prepare request parameters
        params = {
            **self._default_params,
            "messages": formatted_messages,
            **kwargs
        }
        
        if stop:
            params["stop"] = stop
        
        # Log request for debugging
        if self.debug_mode:
            debug_params = params.copy()
            logger.info(f"Using model: {debug_params.get('model')}")
            logger.info(f"Request params: {json.dumps(debug_params, default=str)}")
        
        try:
            # Make API call
            response = requests.post(
                self.openrouter_url,
                headers=headers,
                json=params,
                timeout=self.timeout
            )
            
            # Log API response for debugging
            if self.debug_mode:
                logger.info(f"Response status: {response.status_code}")
                try:
                    logger.info(f"Response JSON: {json.dumps(response.json(), indent=2)}")
                except:
                    logger.info(f"Response text: {response.text}")
            
            # Handle errors
            if response.status_code != 200:
                error_message = f"OpenRouter API returned error {response.status_code}: {response.text}"
                logger.error(error_message)
                raise ValueError(error_message)
            
            # Parse response
            try:
                parsed_response = response.json()
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON response: {response.text}")
                raise ValueError(f"OpenRouter returned invalid JSON: {response.text}")
            
            # Check for expected fields
            if "choices" not in parsed_response:
                logger.error(f"Response missing 'choices' field: {parsed_response}")
                
                if "error" in parsed_response:
                    error_msg = parsed_response.get("error", {}).get("message", "Unknown error")
                    raise ValueError(f"OpenRouter response error: {error_msg}")
                
                # Create a synthetic response
                logger.warning("Creating synthetic response due to missing 'choices' field")
                synthetic_msg = "I encountered an issue with the API response. Please try again."
                
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(content=synthetic_msg),
                            generation_info={"synthetic_response": True}
                        )
                    ],
                    llm_output={"raw_response": parsed_response}
                )
            
            return self._create_chat_result(parsed_response)
            
        except requests.RequestException as e:
            logger.error(f"Request to OpenRouter failed: {str(e)}")
            raise ValueError(f"Error communicating with OpenRouter API: {str(e)}")
    
    def summarize(
        self, 
        text: str,
        custom_prompt: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Summarize a text using the OpenRouter API.
        
        Args:
            text: Text content to summarize
            custom_prompt: Optional custom prompt template
            
        Returns:
            Tuple of (summary_text, metadata)
        """
        # Create default prompt if not provided
        if not custom_prompt:
            custom_prompt = """
            You are a professional summarizer with expertise in creating clear, concise, and comprehensive summaries.

            Please summarize the following text:

            {text}

            Focus on the main points, key arguments, and important details while maintaining the original meaning.
            Structure your summary with a logical flow. Include an introduction, body, and conclusion.
            """
            
        # Replace placeholder with actual text
        prompt = custom_prompt.replace("{text}", text)
        
        # Create message
        messages = [
            SystemMessage(content="You are a professional summarizer."),
            HumanMessage(content=prompt)
        ]
        
        # Generate response
        result = self._generate(messages=messages)
        
        if not result.generations:
            summary = "Unable to generate summary."
            metadata = {"error": "No generations returned"}
        else:
            summary = result.generations[0].message.content
            metadata = {
                "model": self.model_name,
                "finish_reason": result.generations[0].generation_info.get("finish_reason"),
                **result.llm_output
            }
        
        return summary, metadata
    
    def answer_question(
        self,
        document_text: str,
        question: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Answer a question based on document text.
        
        Args:
            document_text: The document text to reference
            question: The question to answer
            
        Returns:
            Tuple of (answer_text, metadata)
        """
        # Create QA prompt
        qa_prompt = f"""
        You are an AI assistant answering questions about documents.
        
        Document text:
        {document_text}
        
        Based only on the information in the document above, please answer the following question:
        {question}
        
        If the answer cannot be found in the document, say "I don't have enough information in the document to answer this question."
        
        Answer:
        """
        
        # Create messages
        messages = [
            SystemMessage(content="You are a helpful assistant that answers questions about documents."),
            HumanMessage(content=qa_prompt)
        ]
        
        # Generate response
        result = self._generate(messages=messages)
        
        if not result.generations:
            answer = "Unable to generate an answer."
            metadata = {"error": "No generations returned"}
        else:
            answer = result.generations[0].message.content
            metadata = {
                "model": self.model_name,
                "finish_reason": result.generations[0].generation_info.get("finish_reason"),
                **result.llm_output
            }
        
        return answer, metadata
    
    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Fetch available models from OpenRouter API."""
        url = "https://openrouter.ai/api/v1/models"
        api_key = os.environ.get("OPENROUTER_API_KEY")
        
        if not api_key:
            logger.error("OpenRouter API key not found in environment variables")
            return []
            
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://document-summarizer-api.example.com/",
                "Content-Type": "application/json"
            }
            
            logger.info("Fetching models from OpenRouter API")
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"Error fetching models: {response.status_code}: {response.text}")
                return []
                
            data = response.json()
            models = data.get("data", [])
            
            logger.info(f"Retrieved {len(models)} models from OpenRouter")
            
            # Format model information
            formatted_models = []
            for model in models:
                # Extract key information from the model data
                model_info = {
                    "name": model.get("id", "unknown"),
                    "description": model.get("description", ""),
                    "context_length": str(model.get("context_length", "")),
                    "cost": model.get("pricing", {}).get("prompt", "")
                }
                
                # Try to extract additional information for better display
                if "training_data" in model:
                    model_info["training_data"] = model.get("training_data")
                
                if "pricing" in model:
                    prompt_price = model.get("pricing", {}).get("prompt", "")
                    completion_price = model.get("pricing", {}).get("completion", "")
                    if prompt_price and completion_price and prompt_price != completion_price:
                        model_info["cost"] = f"{prompt_price}/{completion_price}"
                
                formatted_models.append(model_info)
                
            return formatted_models
            
        except Exception as e:
            logger.error(f"Error fetching OpenRouter models: {str(e)}")
            return []
    
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "openrouter"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

# For backward compatibility
class OpenRouterLLM(ChatOpenRouter):
    """Alias for backward compatibility."""
    pass