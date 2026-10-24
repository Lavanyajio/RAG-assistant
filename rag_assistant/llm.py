"""Thin wrapper around the Hugging Face Inference API chat endpoint."""
import logging
import time
from typing import Optional

from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the language model cannot be reached or returns an error."""


class HFChatModel:
    def __init__(
        self,
        model: str,
        token: str,
        provider: str = "",
        max_tokens: int = 700,
        temperature: float = 0.2,
        client: Optional[object] = None,
        retries: int = 2,
    ):
        self.model = model
        self.token = token
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = retries
        self._client = client

    @property
    def client(self):
        if self._client is None:
            if not self.token:
                raise LLMError("HF_TOKEN is not set. Add it to your .env file (see .env.example).")
            kwargs = {"model": self.model, "token": self.token}
            if self.provider:
                kwargs["provider"] = self.provider
            self._client = InferenceClient(**kwargs)
        return self._client

    def chat(self, messages: list, max_tokens: Optional[int] = None,
             temperature: Optional[float] = None) -> str:
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                response = self.client.chat_completion(
                    messages=messages,
                    max_tokens=max_tokens or self.max_tokens,
                    temperature=self.temperature if temperature is None else temperature,
                )
                return (response.choices[0].message.content or "").strip()
            except LLMError:
                raise
            except Exception as exc:  # network / provider errors
                last_error = exc
                logger.warning("LLM call failed (attempt %s): %s", attempt + 1, exc)
                time.sleep(1.5 * (attempt + 1))
        raise LLMError(f"Model '{self.model}' failed: {last_error}")
