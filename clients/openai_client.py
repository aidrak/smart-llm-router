from openai import OpenAI
from typing import Dict, Any, List, Optional
from clients.base import BaseLLMClient
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

class OpenAIClient(BaseLLMClient):
    def __init__(self):
        super().__init__("openai")
        self.provider_name = "openai"
    
    def initialize_client(self) -> bool:
        if not settings.openai_api_key:
            logger.warning("OPENAI_API_KEY not set. OpenAI models will not be available.")
            return False
        
        try:
            self.client = OpenAI(api_key=settings.openai_api_key)
            logger.info("✅ OpenAI client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            return False
    
    async def generate_response(self, 
                              model_id: str, 
                              messages: List[Dict[str, Any]], 
                              temperature: float,
                              api_parameters: Dict[str, Any],
                              system_prompt: Optional[str] = None) -> Dict[str, Any]:
        
        llm_messages = []
        if system_prompt:
            llm_messages.append({"role": "system", "content": system_prompt})
        
        for msg in messages:
            # The content can be a string or a list of dicts (for multimodal)
            # The OpenAI API expects the content in this format directly.
            llm_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        call_params = {
            "model": model_id,
            "messages": llm_messages
        }
        
        if not api_parameters.get("exclude_temperature", False):
            call_params["temperature"] = temperature
        
        custom_flags = {"exclude_temperature"}
        for param_name, param_value in api_parameters.items():
            if param_name not in custom_flags:
                call_params[param_name] = param_value
        
        try:
            response = self.client.chat.completions.create(**call_params)
            logger.info(f"✅ Received response from OpenAI {model_id}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error calling OpenAI {model_id}: {e}")
            raise
