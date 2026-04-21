"""LLM wrappers that capture reasoning/thinking traces.

Two providers are supported, both returning the normal `generations`
field so the AutoReason tournament is unchanged:

  * `ReasoningOpenRouterLLM` — OpenRouter reasoning-capable models
    (minimax m2.x, deepseek-r1). Pulls `choices[0].message.reasoning`.

  * `LocalInlineThinkLLM` — any OpenAI-compatible server that emits
    Qwen3-style inline `<think>…</think>` reasoning in the content.
    Works with mlx-lm's server, vLLM, Ollama with Qwen3, etc.
    Strips the think block out of the returned text and stores it in
    the reasoning buffer.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field, PrivateAttr

from distilabel.models.llms.openai import OpenAILLM


_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL | re.IGNORECASE)


def _split_think(text: str) -> Tuple[str, str]:
    """Return (reasoning, stripped_text) from a content string that may
    contain one or more <think>...</think> blocks. Reasoning is joined
    with blank lines; stripped_text is the content with think blocks removed.
    If no think tag is present, returns ("", text).
    """
    if not text:
        return "", text
    thoughts: List[str] = []

    def _collect(m):
        thoughts.append(m.group(1).strip())
        return ""

    stripped = _THINK_RE.sub(_collect, text).strip()
    reasoning = "\n\n".join(thoughts)
    return reasoning, stripped


class ReasoningOpenRouterLLM(OpenAILLM):
    """OpenAILLM that also captures OpenRouter `reasoning` output.

    Usage::

        llm = ReasoningOpenRouterLLM(
            model="minimax/minimax-m2.7",
            base_url="https://openrouter.ai/api/v1",
            api_key="...",
            reasoning_effort="high",
        )
        llm.load()
        # run a tournament ...
        captured = llm.drain_reasoning()  # List[(role_hint, reasoning, text)]
    """

    reasoning_effort: str = Field(
        default="high",
        description="OpenRouter reasoning effort level: 'high', 'medium', 'low'.",
    )
    reasoning_max_tokens: Optional[int] = Field(
        default=None,
        description="Optional explicit reasoning token budget (OpenRouter).",
    )

    # Private buffer — populated by agenerate, drained by caller per sample.
    _reasoning_buffer: List[Tuple[str, str, str]] = PrivateAttr(default_factory=list)

    def _role_hint_from_input(self, input_msgs) -> str:
        # Best-effort: inspect first system message content for a role signature.
        try:
            first = (input_msgs[0].get("content") or "")[:120].lower()
        except Exception:  # noqa: BLE001
            return "unknown"
        if "rigorous critic" in first or first.startswith("rigorous critic"):
            return "critic"
        if "adversarial reviser" in first:
            return "author_b"
        if "conservative synthesizer" in first:
            return "synthesizer"
        if "impartial judge" in first:
            return "judge"
        if "expert assistant" in first or "careful, helpful expert" in first:
            return "teacher"
        return "unknown"

    def drain_reasoning(self) -> List[Tuple[str, str, str]]:
        out = list(self._reasoning_buffer)
        self._reasoning_buffer.clear()
        return out

    async def agenerate(  # type: ignore[override]
        self,
        input: Any,
        num_generations: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Build the reasoning request block per OpenRouter spec.
        reasoning_block: Dict[str, Any] = {"effort": self.reasoning_effort}
        if self.reasoning_max_tokens is not None:
            reasoning_block["max_tokens"] = self.reasoning_max_tokens

        # Merge into extra_body / generation_kwargs — OpenAI SDK threads
        # `extra_body` into the raw JSON payload.
        extra_body = dict(kwargs.pop("extra_body", None) or {})
        extra_body.setdefault("reasoning", reasoning_block)
        kwargs["extra_body"] = extra_body

        # Get the raw completion directly so we can access `.reasoning`.
        # Split generation_kwargs: OpenAI-standard fields go top-level;
        # extra_body (server-specific extensions like repetition_penalty or
        # chat_template_kwargs) gets merged with any per-call extra_body kwarg.
        gen = dict(self.generation_kwargs or {})  # type: ignore[arg-type]
        gen_extra_body = dict(gen.pop("extra_body", None) or {})
        call_extra_body = dict(kwargs.pop("extra_body", None) or {})
        merged_extra_body = {**gen_extra_body, **call_extra_body}
        if merged_extra_body:
            kwargs["extra_body"] = merged_extra_body

        client = self._aclient  # type: ignore[attr-defined]
        completion = await client.chat.completions.create(
            model=self.model,
            messages=input,
            n=num_generations,
            **gen,
            **kwargs,
        )

        generations: List[str] = []
        input_tokens = [0] * num_generations
        output_tokens = [0] * num_generations
        role_hint = self._role_hint_from_input(input)

        for i, choice in enumerate(completion.choices):
            msg = choice.message
            text = (msg.content or "") if msg else ""
            reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
            generations.append(text)
            self._reasoning_buffer.append((role_hint, reasoning or "", text))

        # Populate token statistics if usage is present.
        usage = getattr(completion, "usage", None)
        if usage is not None:
            if getattr(usage, "prompt_tokens", None) is not None:
                input_tokens = [usage.prompt_tokens] * num_generations
            if getattr(usage, "completion_tokens", None) is not None:
                output_tokens = [usage.completion_tokens] * num_generations

        return {
            "generations": generations,
            "statistics": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }


class LocalInlineThinkLLM(OpenAILLM):
    """OpenAI-compatible client for local servers that emit `<think>…</think>`
    inline (Qwen3-family default). Strips the think block from the returned
    generation and stores it in a role-keyed reasoning buffer.

    Does NOT request any extra_body — many local servers reject unknown
    fields. Works out of the box with mlx-lm server, vLLM, Ollama.
    """

    _reasoning_buffer: List[Tuple[str, str, str]] = PrivateAttr(default_factory=list)

    def _role_hint_from_input(self, input_msgs) -> str:
        try:
            first = (input_msgs[0].get("content") or "")[:120].lower()
        except Exception:  # noqa: BLE001
            return "unknown"
        if "rigorous critic" in first or first.startswith("rigorous critic"):
            return "critic"
        if "adversarial reviser" in first:
            return "author_b"
        if "conservative synthesizer" in first:
            return "synthesizer"
        if "impartial judge" in first:
            return "judge"
        if "expert assistant" in first or "careful, helpful expert" in first:
            return "teacher"
        return "unknown"

    def drain_reasoning(self) -> List[Tuple[str, str, str]]:
        out = list(self._reasoning_buffer)
        self._reasoning_buffer.clear()
        return out

    async def agenerate(  # type: ignore[override]
        self,
        input: Any,
        num_generations: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Split generation_kwargs: OpenAI-standard fields go top-level;
        # extra_body (server-specific extensions like repetition_penalty or
        # chat_template_kwargs) gets merged with any per-call extra_body kwarg.
        gen = dict(self.generation_kwargs or {})  # type: ignore[arg-type]
        gen_extra_body = dict(gen.pop("extra_body", None) or {})
        call_extra_body = dict(kwargs.pop("extra_body", None) or {})
        merged_extra_body = {**gen_extra_body, **call_extra_body}
        if merged_extra_body:
            kwargs["extra_body"] = merged_extra_body

        client = self._aclient  # type: ignore[attr-defined]
        completion = await client.chat.completions.create(
            model=self.model,
            messages=input,
            n=num_generations,
            **gen,
            **kwargs,
        )

        generations: List[str] = []
        role_hint = self._role_hint_from_input(input)
        for choice in completion.choices:
            msg = choice.message
            raw = (msg.content or "") if msg else ""
            reasoning_inline, stripped = _split_think(raw)
            # Also pick up a separate reasoning field if the server provides one.
            sep_reasoning = (
                getattr(msg, "reasoning", None)
                or getattr(msg, "reasoning_content", None)
                or ""
            )
            reasoning = (sep_reasoning + "\n\n" + reasoning_inline).strip() if sep_reasoning else reasoning_inline
            # Gemma-family thinking format sometimes puts the final answer at
            # the tail of reasoning_content and leaves content empty. Fall back
            # so the tournament sees real text to critique/rank.
            text_for_tournament = stripped if stripped else reasoning
            generations.append(text_for_tournament)
            self._reasoning_buffer.append((role_hint, reasoning, text_for_tournament))

        usage = getattr(completion, "usage", None)
        input_tokens = [0] * num_generations
        output_tokens = [0] * num_generations
        if usage is not None:
            if getattr(usage, "prompt_tokens", None) is not None:
                input_tokens = [usage.prompt_tokens] * num_generations
            if getattr(usage, "completion_tokens", None) is not None:
                output_tokens = [usage.completion_tokens] * num_generations

        return {
            "generations": generations,
            "statistics": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }
