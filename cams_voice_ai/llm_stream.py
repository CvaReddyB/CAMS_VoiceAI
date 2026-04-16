"""OpenAI Chat Completions streaming with optional usage on the stream."""

from __future__ import annotations

import logging
from typing import Callable, Iterator, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


def stream_assistant_text(
    *,
    api_key: str,
    model: str,
    messages: List[dict],
    temperature: float = 0.45,
    include_usage: bool = True,
    reasoning_effort: Optional[str] = None,
    on_delta: Optional[Callable[[str], None]] = None,
) -> Iterator[str]:
    client = OpenAI(api_key=api_key)
    kwargs = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
    }
    if include_usage:
        kwargs["stream_options"] = {"include_usage": True}
    if reasoning_effort and "gpt-5" in model.lower():
        kwargs["reasoning_effort"] = reasoning_effort

    try:
        stream = client.chat.completions.create(**kwargs)
    except TypeError as exc:
        logger.debug("OpenAI stream retry without optional params: %s", exc)
        kwargs.pop("stream_options", None)
        kwargs.pop("reasoning_effort", None)
        stream = client.chat.completions.create(**kwargs)
    for chunk in stream:
        if getattr(chunk, "usage", None) is not None and include_usage:
            u = chunk.usage
            if u is not None:
                logger.debug(
                    "OpenAI stream usage: prompt=%s completion=%s total=%s",
                    getattr(u, "prompt_tokens", None),
                    getattr(u, "completion_tokens", None),
                    getattr(u, "total_tokens", None),
                )
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        piece = getattr(delta, "content", None) if delta is not None else None
        if not piece:
            continue
        if on_delta:
            on_delta(piece)
        yield piece
