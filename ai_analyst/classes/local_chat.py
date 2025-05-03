from ai_analyst.general_utils.text_utils import _txt
from ai_analyst.utils.inference import analyst_inference
from ai_analyst.analysis_kit.config import AnalysisConfig

class _LLMResponse:
    def __init__(self, text: str):
        self.text = text

class LocalChat:
    """Stateless Gemma‑3 backend mimicking google.genai.Chat"""
    def __init__(self, system_prompt: str = "You are an AI assistant."):
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": _txt(system_prompt)})
        self.max_history = 3  # Keep only the last 3 messages

    def send_message(self, user_message, config: AnalysisConfig = None):
        if isinstance(user_message, str):
            self.messages.append({"role": "user", "content": _txt(user_message)})
        elif isinstance(user_message, list):
            self.messages.extend(user_message)
        else:
            raise TypeError("user_message must be str or list[dict]")

        # Keep only the last max_history messages
        if len(self.messages) > self.max_history:
            # Always keep the system message
            system_msg = self.messages[0] if self.messages[0]["role"] == "system" else None
            self.messages = self.messages[-self.max_history:]
            if system_msg and system_msg not in self.messages:
                self.messages.insert(0, system_msg)

        if config is None:
            config = AnalysisConfig()
            
        assistant_reply = analyst_inference(self.messages, config=config)
        self.messages.append({"role": "assistant", "content": _txt(assistant_reply)})
        return _LLMResponse(assistant_reply) 