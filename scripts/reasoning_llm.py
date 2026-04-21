"""OpenRouter LLM wrapper that captures reasoning/thinking traces.

Reasoning-capable models on OpenRouter (minimax m2.x, deepseek-r1, etc.)
emit a separate `reasoning` field in the assistant message. distilabel's
OpenAILLM drops it. This subclass:

  * Requests reasoning via extra_body `{"reasoning": {"effort": "high"}}`.
  * Captures the reasoning text on each agenerate() call.
  * Appends (role_hint, reasoning_text, final_text) tuples to a buffer
    that can be read / reset between samples.

The tournament uses the normal `generations` field as before — reasoning
is sidecar data threaded via the buffer, so no changes to tournament.py.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field, PrivateAttr

from distilabel.models.llms.openai import OpenAILLM


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
        if "rigorous critic" in first:
            return "critic"
        if "adversarial reviser" in first:
            return "author_b"
        if "conservative synthesizer" in first:
            return "synthesizer"
        if "impartial judge" in first:
            return "judge"
        if "careful, helpful expert" in first:
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
        client = self._aclient  # type: ignore[attr-defined]
        completion = await client.chat.completions.create(
            model=self.model,
            messages=input,
            n=num_generations,
            **{k: v for k, v in self.generation_kwargs.items() if k != "extra_body"},  # type: ignore[attr-defined]
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
