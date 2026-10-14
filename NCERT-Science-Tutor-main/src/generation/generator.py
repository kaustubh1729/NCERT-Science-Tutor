"""Answer generation against an OpenAI-compatible endpoint.

## Why an 8B model is the right size here, not a compromise

The generator's job in this system is deliberately small. Retrieval and reranking
have already located the passage that contains the answer; generation only has to
read five passages and restate what they say in a fixed structure. That is an
extraction and formatting task, not a reasoning one — and the quality ceiling is set
by the retriever, not the LLM. Spending 10x per token on a frontier model buys very
little on an extractive task and buys *nothing* when the retrieved context is wrong.

The corollary is that generation failures are usually retrieval failures wearing a
disguise, which is why ``docs/notes/evaluation.md`` separates the two in the metrics.

## Why NVIDIA NIM behind the OpenAI SDK

NIM speaks the OpenAI chat-completions schema. Using the OpenAI client against it
means the provider is a base-URL change, not a rewrite — so migrating to Groq,
Together, OpenRouter or a local vLLM is a config edit. Vendor independence in an
LLM app is cheap if you buy it at the start and expensive if you retrofit it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from langchain_core.documents import Document
from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from src.config import GenerationConfig, get_settings
from src.generation.prompt import PromptBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a careful NCERT Science textbook assistant for students in classes 6-10. "
    "You answer strictly from the textbook passages you are given. "
    "You never introduce facts from outside those passages, and you say plainly when "
    "the passages do not contain the answer."
)


class GenerationError(RuntimeError):
    """Generation failed in a way the caller should surface to the user."""


class NCERTGenerator:
    """Wraps the chat-completions call and the prompt contract."""

    def __init__(self, api_key: str | None = None, config: GenerationConfig | None = None):
        self.config = config or get_settings().generation
        self.prompt_builder = PromptBuilder()

        key = api_key or os.getenv(self.config.api_key_env)
        if not key:
            raise GenerationError(
                f"{self.config.api_key_env} is not set. The application cannot generate "
                f"answers without it. Set it in your .env file locally, or in the Render "
                f"dashboard under Environment."
            )

        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=key,
            timeout=self.config.timeout_s,
            # Retries only fire on connection errors and 5xx/429 — never on a 400,
            # where retrying an identical malformed request just burns quota.
            max_retries=self.config.max_retries,
        )

    def _messages(self, query: str, documents: list[Document]) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.prompt_builder.build(query, documents)},
        ]

    def generate_answer(self, query: str, documents: list[Document]) -> str:
        """Produce a grounded answer.

        Returns the refusal string without calling the model when there is no
        context — asking an LLM to answer from nothing is the most reliable way to
        get a hallucination, and it costs a network round trip to obtain one.
        """
        if not documents:
            logger.warning("no context retrieved for %r; refusing without calling the model", query)
            return self.prompt_builder.NO_CONTEXT_ANSWER

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=self._messages(query, documents),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except APITimeoutError as exc:
            raise GenerationError(
                f"The language model did not respond within {self.config.timeout_s:.0f}s. "
                f"This is usually transient — please try again."
            ) from exc
        except RateLimitError as exc:
            raise GenerationError(
                "The language model API rate limit was reached. Wait a moment and retry."
            ) from exc
        except APIError as exc:
            logger.error("LLM API error: %s", exc)
            raise GenerationError(f"The language model returned an error: {exc}") from exc

        content = response.choices[0].message.content
        if not content or not content.strip():
            logger.warning("model returned empty content for %r", query)
            return self.prompt_builder.NO_CONTEXT_ANSWER

        if response.usage:
            logger.info(
                "generated %d completion tokens from %d prompt tokens",
                response.usage.completion_tokens,
                response.usage.prompt_tokens,
            )
        return content.strip()

    def stream_answer(self, query: str, documents: list[Document]) -> Iterator[str]:
        """Yield the answer incrementally.

        Generation is ~12 s of the ~15 s end-to-end time, and a blocking spinner for
        that long reads as a hang. Streaming does not make it faster; it makes the
        wait legible, which is the part users actually judge.
        """
        if not documents:
            yield self.prompt_builder.NO_CONTEXT_ANSWER
            return

        try:
            stream = self.client.chat.completions.create(
                model=self.config.model,
                messages=self._messages(query, documents),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except (APITimeoutError, RateLimitError, APIError) as exc:
            logger.error("streaming generation failed: %s", exc)
            raise GenerationError(f"The language model returned an error: {exc}") from exc
