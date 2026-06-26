import logging
from typing import List, Dict, Optional

from src.config import MAX_MEMORY_TURNS

logger = logging.getLogger(__name__)


class ConversationMemory:
    def __init__(self, max_turns: int = MAX_MEMORY_TURNS):
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []

    def add_turn(self, user: str, assistant: str):
        self.history.append({"role": "user", "content": user})
        self.history.append({"role": "assistant", "content": assistant})
        while len(self.history) > self.max_turns * 2:
            self.history.pop(0)
        logger.debug(f"对话历史: {len(self.history)} 条消息")

    def get_history(self) -> List[Dict[str, str]]:
        return list(self.history)

    def get_history_text(self, max_chars: int = 2000) -> str:
        texts = []
        total = 0
        for msg in reversed(self.history):
            text = f"{'用户' if msg['role'] == 'user' else 'AI'}: {msg['content']}"
            if total + len(text) > max_chars:
                break
            texts.insert(0, text)
            total += len(text)
        return "\n".join(texts) if texts else ""

    def get_last_user_message(self) -> Optional[str]:
        for msg in reversed(self.history):
            if msg["role"] == "user":
                return msg["content"]
        return None

    def extract_context_keywords(self) -> List[str]:
        keywords = []
        for msg in reversed(self.history):
            if msg["role"] == "user":
                keywords.append(msg["content"])
                if len(keywords) >= 3:
                    break
        return keywords

    def clear(self):
        self.history.clear()
        logger.info("对话历史已清空")

    def size(self) -> int:
        return len(self.history) // 2
