import os
import time
import json
from typing import List, Dict, Optional

import requests
from dotenv import load_dotenv


load_dotenv()  # load .env if present
# Try loading .sample.env as a fallback (for convenience in this repo)
try:
    from pathlib import Path
    sample_env = Path(__file__).resolve().parents[2] / ".sample.env"
    if sample_env.exists():
        load_dotenv(dotenv_path=sample_env, override=False)
except Exception:
    pass


class BaseModel:
    """Abstract base for all models."""

    def generate(self, prompt: str, temperature: float = 0.7, system: Optional[str] = None) -> str:
        raise NotImplementedError

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        raise NotImplementedError


class ChatCompletionsModel(BaseModel):
    """Minimal OpenAI-compatible Chat Completions client.

    Works with DeepSeek (default) and local OpenAI-compatible servers like LM Studio.
    Configure via env:
      - LLM_API_BASE (e.g., http://127.0.0.1:1234)
      - LLM_API_KEY (optional for local)
      - Fallbacks: DEEPSEEK_API_BASE, DEEPSEEK_API_KEY
    """

    def __init__(self, model_name: str = "deepseek-chat", api_base: str | None = None, api_key: str | None = None):
        self.model_name = model_name or "deepseek-chat"
        self.api_base = (
            api_base
            or os.getenv("LLM_API_BASE")
            or os.getenv("DEEPSEEK_API_BASE")
            or "https://api.deepseek.com"
        )
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")

    def _post(self, path: str, payload: dict, retries: int = 3, backoff: float = 1.0) -> dict:
        url = f"{self.api_base}/v1{path}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_err = None
        for attempt in range(retries):
            try:
                resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
                if resp.status_code == 200:
                    return resp.json()
                # Handle rate limits and transient errors
                if resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(backoff * (2 ** attempt))
                    continue
                # Other errors: raise with details
                try:
                    detail = resp.json()
                except Exception:
                    detail = {"text": resp.text}
                raise RuntimeError(f"Chat API error {resp.status_code}: {detail}")
            except Exception as e:
                last_err = e
                time.sleep(backoff * (2 ** attempt))
        if last_err:
            raise last_err
        raise RuntimeError("Chat API request failed without specific error")

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        data = self._post("/chat/completions", payload)
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Unexpected Chat API response format: {data}") from e

    def generate(self, prompt: str, temperature: float = 0.7, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, temperature=temperature)


def create_model(model_name: str) -> BaseModel:
    """Factory function.

    Defaults to DeepSeek if no env is provided, otherwise respects LLM_API_BASE
    for local OpenAI-compatible servers (e.g., LM Studio).
    """
    model = model_name or "deepseek-chat"
    return ChatCompletionsModel(model)
