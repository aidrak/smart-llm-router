# clients/anthropic_client.py
from anthropic import Anthropic
from typing import Dict, Any, List, Optional
from clients.base import BaseLLMClient
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

class AnthropicClient(BaseLLMClient):
    def __init__(self):
        super().__init__("anthropic")
    
    def initialize_client(self) -> bool:
        if not settings.anthropic_api_key:
            logger.warning("ANTHROPIC_API_KEY not set. Anthropic models will not be available.")
            return False
        
        try:
            self.client = Anthropic(api_key=settings.anthropic_api_key)
            logger.info("✅ Anthropic client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Anthropic client: {e}")
            return False
    
    async def generate_response(self, model_id: str, messages: List[Dict[str, Any]], 
                              temperature: float, api_parameters: Dict[str, Any],
                              system_prompt: Optional[str] = None) -> Dict[str, Any]:
        
        # Prepare messages (Anthropic doesn't use system messages in the messages array)
        anthropic_messages = []
        for msg in messages:
            if msg.get("role") != "system":
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Build call parameters
        call_params = {
            "model": model_id,
            "messages": anthropic_messages
        }
        
        # Add temperature unless excluded
        if not api_parameters.get("exclude_temperature", False):
            call_params["temperature"] = temperature
        
        # Add system prompt if available
        if system_prompt:
            call_params["system"] = system_prompt
        
        # Add other parameters
        custom_flags = {"exclude_temperature"}
        for param_name, param_value in api_parameters.items():
            if param_name not in custom_flags:
                call_params[param_name] = param_value
        
        # Set default max_tokens if not specified
        if "max_tokens" not in call_params:
            call_params["max_tokens"] = 4096
        
        try:
            response = self.client.messages.create(**call_params)
            logger.info(f"✅ Received response from Anthropic {model_id}")
            
            return {
                "choices": [{"message": {"content": response.content[0].text}}],
                "model": model_id,
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            }
        except Exception as e:
            logger.error(f"❌ Error calling Anthropic {model_id}: {e}")
            raise