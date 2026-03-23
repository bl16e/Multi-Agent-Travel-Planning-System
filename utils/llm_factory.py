from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from utils.settings import get_settings


@lru_cache(maxsize=1)
def build_qwen_chat() -> ChatOpenAI | None:
    settings = get_settings()
    if not settings.qwen_api_key:
        return None
    return ChatOpenAI(
        model=settings.qwen_model,
        api_key=settings.qwen_api_key,
        base_url=settings.qwen_base_url,
        temperature=0.2,
        
    )
