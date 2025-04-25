class _LLMResponse:
    def __init__(self, text: str):
        self.text = text

class LocalChat:
    """Stateless Gemma‑3 backend mimicking google.genai.Chat"""
    def __init__(self, system_prompt: str = "You are an AI assistant."):
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": _txt(system_prompt)})

    def send_message(self, user_message):
        if isinstance(user_message, str):
            self.messages.append({"role": "user", "content": _txt(user_message)})
        elif isinstance(user_message, list):
            self.messages.extend(user_message)
        else:
            raise TypeError("user_message must be str or list[dict]")

        assistant_reply = analyst_inference(self.messages)
        self.messages.append({"role": "assistant", "content": _txt(assistant_reply)})
        return _LLMResponse(assistant_reply)

class _DummyClient:
    class _Chats:
        @staticmethod
        def create(model: str | None = None):
            return LocalChat()
    chats = _Chats()