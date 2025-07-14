# clients/ollama_client.py
from ollama import Client as OllamaClient as OllamaSDK
from typing import Dict, Any, List, Optional
from clients.base import BaseLLMClient
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

class OllamaClient(BaseLLMClient):
    def __init__(self):
        super().__init__("ollama")
    
    def initialize_client(self) -> bool:
        if not settings.ollama_host:
            logger.warning("OLLAMA_API_HOST not set. Ollama models will not be available.")
            return False
        
        try:
            self.client = OllamaSDK(host=settings.ollama_host)
            logger.info("✅ Ollama client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Ollama client: {e}")
            return False
    
    async def generate_response(self, model_id: str, messages: List[Dict[str, Any]], 
                              temperature: float, api_parameters: Dict[str, Any],
                              system_prompt: Optional[str] = None) -> Dict[str, Any]:
        # Prepare messages
        llm_messages = []
        if system_prompt:
            llm_messages.append({"role": "system", "content": system_prompt})
        
        for msg in messages:
            if msg.get("role") != "system":
                llm_messages.append(msg)
        
        # Build parameters
        call_params = {
            "model": model_id,
            "messages": llm_messages,
            "stream": False
        }
        
        # Handle options
        ollama_options = api_parameters.get('options', {})
        if not api_parameters.get("exclude_temperature", False):
            ollama_options["temperature"] = temperature
        
        if ollama_options:
            call_params["options"] = ollama_options
        
        # Add other parameters
        custom_flags = {"exclude_temperature", "options"}
        for param_name, param_value in api_parameters.items():
            if param_name not in custom_flags:
                call_params[param_name] = param_value
        
        try:
            response = self.client.chat(**call_params)
            logger.info(f"✅ Received response from Ollama {model_id}")
            return response
        except Exception as e:
            logger.error(f"❌ Error calling Ollama {model_id}: {e}")
            raise