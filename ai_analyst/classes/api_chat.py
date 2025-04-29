from google import genai
from ai_analyst.analysis_kit.config import AnalysisConfig
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class APIResponse:
    """Response object for API chat to match local client interface."""
    text: str
    model: str
    config: AnalysisConfig


class APIChat:
    """API-based chat client using Google's genai package.
    
    This class provides a drop-in replacement for the local chat client,
    using Google's API for model inference instead of running locally.
    """
    
    def __init__(self, config: AnalysisConfig):
        """Initialize the API client.
        
        Args:
            config: Configuration object containing API settings
        """
        self.config = config
        self.client = genai.Client(api_key=config.api_key)
        self.model_id = config.api_model_id
        self.chats = self  # To match the interface of the local client
        
    def create(self, model: str):
        """Create a new chat session.
        
        Args:
            model: Model ID to use
            
        Returns:
            self: Returns self to allow method chaining
        """
        self.model_id = model
        return self
        
    def send_message(self, message: str, config: Optional[AnalysisConfig] = None) -> APIResponse:
        """Send a message to the API and get the response.
        
        Args:
            message: The message to send
            config: Optional configuration override
            
        Returns:
            APIResponse object containing the response text and metadata
        """
        # Use provided config or default to instance config
        config = config or self.config
        
        # Create a chat session
        chat = self.client.chats.create(model=self.model_id)
        
        # Send the message and get the response
        response = chat.send_message(message)
        
        # Return a response object that matches the local client's interface
        return APIResponse(
            text=response.text,
            model=self.model_id,
            config=config
        ) 