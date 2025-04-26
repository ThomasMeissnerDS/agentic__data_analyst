from ai_analyst.classes.local_chat import LocalChat

class _DummyClient:
    class _Chats:
        @staticmethod
        def create(model: str | None = None):
            return LocalChat()
    chats = _Chats() 