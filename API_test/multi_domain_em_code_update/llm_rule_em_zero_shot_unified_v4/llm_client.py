from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import requests


@dataclass
class LLMConfig:
    provider: str
    model: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0
    timeout: int = 120
    max_tokens: int = 2048


class BaseLLMClient:
    def chat(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError


class OllamaClient(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_base = config.api_base or "http://127.0.0.1:11434"

    def chat(self, messages: List[Dict[str, str]]) -> str:
        url = self.api_base.rstrip("/") + "/api/chat"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "seed": 42,
            },
        }
        r = requests.post(url, json=payload, timeout=self.config.timeout)
        try:
            r.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"Ollama request failed: {r.status_code} {r.text}") from exc
        data = r.json()
        return data["message"]["content"].strip()


class OpenAICompatibleClient(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        self.config = config
        if not config.api_base:
            raise ValueError("OpenAI-compatible provider requires api_base")
        if not config.api_key:
            raise ValueError("OpenAI-compatible provider requires api_key")

    @staticmethod
    def _extract_content(message: Any) -> str:
        if not isinstance(message, dict):
            return str(message or "").strip()

        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif "text" in item:
                        parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "\n".join(p for p in parts if p).strip()

        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()

        return str(content).strip()

    def chat(self, messages: List[Dict[str, str]]) -> str:
        url = self.config.api_base.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }

        r = requests.post(url, headers=headers, json=payload, timeout=self.config.timeout)
        try:
            r.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI-compatible request failed: {r.status_code} {r.text}"
            ) from exc

        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"No choices returned: {data}")

        message = choices[0].get("message", {})
        text = self._extract_content(message)
        if not text:
            raise RuntimeError(f"Empty content returned: {data}")
        return text


def build_client(config: LLMConfig) -> BaseLLMClient:
    if config.provider.lower() == "ollama":
        return OllamaClient(config)
    if config.provider.lower() in {"openai_compatible", "openai-compatible"}:
        return OpenAICompatibleClient(config)
    raise ValueError(f"Unsupported provider: {config.provider}")
