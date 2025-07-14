from typing import List, Dict, Any
from clients.base import get_llm_client_and_model_details
from config.settings import settings
from utils.logger import setup_logger
import json

logger = setup_logger(__name__)

class MessageClassifier:
    def __init__(self):
        pass

    def _get_text_from_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(part.get("text", "") for part in content if part.get("type") == "text")
        return ""

    def is_title_generation_request(self, messages: List[Dict[str, Any]]) -> bool:
        if not messages:
            return False
        
        last_message_content = self._get_text_from_content(messages[-1].get("content", "")).lower()
        
        title_patterns = [
            "generate a title", "create a title", "title for this conversation",
            "chat_history", "</chat_history>", "conversation title",
            "summarize this conversation into a title"
        ]
        
        return len(last_message_content) < 200 and any(pattern in last_message_content for pattern in title_patterns)

    def is_escalation_request(self, user_prompt: str) -> bool:
        cleaned_prompt = user_prompt.strip().lower()
        return cleaned_prompt in ["escalate", "escalate this", "escalate me"]

    def is_research_request(self, user_prompt: str) -> bool:
        """Detect explicit research requests - more precise detection"""
        cleaned_prompt = user_prompt.strip().lower()
        
        # Imperative research commands (user asking AI to research)
        imperative_triggers = [
            "research this", "research that", "research it", "research about",
            "research the", "research how", "research what", "research when", 
            "research where", "research why", "research whether",
            "look this up", "look that up", "look it up", "look up",
            "search for this", "search for that", "search for the",
            "find information about", "find info about", "find out about",
            "investigate this", "investigate that", "investigate the", 
            "dig deeper into", "get more info on", "get information on"
        ]
        
        # Polite research requests (can you, could you, please, etc.)
        polite_triggers = [
            "can you research", "could you research", "would you research",
            "please research", "can you look up", "could you look up",
            "can you find", "could you find", "can you search",
            "could you search", "please look up", "please find"
        ]
        
        # Question-based research triggers (asking for latest/current info)
        question_triggers = [
            "what's the latest on", "what is the latest on",
            "what's the current", "what is the current", 
            "what are the recent", "what are recent",
            "any recent", "any latest", "any new",
            "current information about", "latest news on", "recent developments"
        ]
        
        # Check if message STARTS with these phrases (more precise)
        for trigger in imperative_triggers + polite_triggers:
            if cleaned_prompt.startswith(trigger):
                return True
        
        # Check for question triggers anywhere in short messages
        if len(cleaned_prompt) < 100:  # Only for short, focused questions
            for trigger in question_triggers:
                if trigger in cleaned_prompt:
                    return True
        
        # Special case: single word "research" followed by a topic
        words = cleaned_prompt.split()
        if len(words) >= 2 and words[0] == "research" and len(words) <= 10:
            return True
        
        return False

    def extract_research_topic(self, user_prompt: str, conversation_context: str) -> str:
        """Extract what to research from the prompt and recent context"""
        prompt_lower = user_prompt.strip().lower()
        
        # If user says "research this" or "research it", we need context
        context_triggers = ["research this", "research that", "research it", "look this up", "look that up", "look it up"]
        
        for trigger in context_triggers:
            if trigger in prompt_lower:
                # Extract the last few topics from conversation context
                context_words = conversation_context.split()
                
                # Look for capitalized words (likely proper nouns/topics)
                potential_topics = []
                for word in context_words[-50:]:  # Last 50 words
                    if len(word) > 2 and (word[0].isupper() or word.replace(',', '').replace('.', '').istitle()):
                        potential_topics.append(word.strip('.,!?;:'))
                
                if potential_topics:
                    return " ".join(potential_topics[-3:])  # Last 3 topics mentioned
                else:
                    return conversation_context[-200:]  # Fallback to recent context
        
        # If explicit topic mentioned, extract it
        if "research about" in prompt_lower:
            return user_prompt.split("research about", 1)[1].strip()
        elif "find information about" in prompt_lower:
            return user_prompt.split("find information about", 1)[1].strip()
        elif "what's the latest on" in prompt_lower:
            return user_prompt.split("what's the latest on", 1)[1].strip()
        
        # Default: return the prompt itself (minus the trigger word)
        words = user_prompt.split()
        if len(words) > 1 and words[0].lower() == "research":
            return " ".join(words[1:])  # Everything after "research"
        
        return user_prompt

    async def classify_message(self, messages: List[Dict[str, Any]]) -> str:
        if not messages:
            return "simple_no_research"
        
        user_content = self._get_text_from_content(messages[-1].get("content", ""))
        
        client, model_id, _, _, system_prompt = get_llm_client_and_model_details(settings.classifier_model)
        
        if not client or not model_id:
            logger.warning(f"Classifier model '{settings.classifier_model}' not available. Using default.")
            return "simple_no_research"
        
        if not system_prompt:
            system_prompt = (
                "Classify the user's request based on two dimensions: 'SIMPLE' or 'HARD' complexity, "
                "and 'RESEARCH' or 'NO_RESEARCH' for information retrieval. "
                "Combine these with an underscore, e.g., 'SIMPLE_RESEARCH'. "
                "Provide ONLY this combined classification. Default to 'SIMPLE_NO_RESEARCH'."
            )
        
        try:
            response = await client.generate_response(
                model_id=model_id,
                messages=[{"role": "user", "content": user_content}],
                temperature=0.0,
                api_parameters={"max_tokens": 25},
                system_prompt=system_prompt
            )
            
            if hasattr(response, 'choices') and response.choices:
                category = response.choices[0].message.content.strip().lower()
            else:
                category = "simple_no_research"
            
            return self._normalize_category(category)
            
        except Exception as e:
            logger.error(f"Error classifying prompt with {settings.classifier_model}: {e}")
            return "simple_no_research"

    def _normalize_category(self, category: str) -> str:
        category = category.strip().lower()
        
        if category.startswith('{'):
            try:
                data = json.loads(category)
                category = data.get('category', 'simple_no_research')
            except json.JSONDecodeError:
                logger.warning(f"Could not decode JSON from classifier: {category}")
                return "simple_no_research"

        valid_categories = ["simple_no_research", "simple_research", "hard_no_research", "hard_research"]
        
        for valid_cat in valid_categories:
            if valid_cat in category:
                return valid_cat
        
        logger.warning(f"Unknown category '{category}' from classifier. Using default.")
        return "simple_no_research"

classifier = MessageClassifier()
