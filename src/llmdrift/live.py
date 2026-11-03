"""Ask every configured model one prompt (used by ``llmdrift ask`` and the dashboard)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from llmdrift.config import GenerationConfig, ModelConfig, PriceEntry
from llmdrift.datasets.schema import Category, GoldenItem
from llmdrift.metrics.cost import compute_cost
from llmdrift.metrics.deterministic import is_abstention, is_refusal
from llmdrift.prompting import build_request
from llmdrift.providers.base import ProviderError
from llmdrift.providers.registry import build_provider, check_available


@dataclass(frozen=True)
class LiveAnswer:
    """One model's answer to an ad-hoc prompt (or why there is none)."""

    model_name: str
    model_id: str
    text: str | None = None
    error: str | None = None
    skipped: str | None = None
    latency_ms: float | None = None
    ttft_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    billed_usd: float | None = None
    list_usd: float | None = None
    refused: bool = False
    abstained: bool = False


def adhoc_item(prompt: str, context: str | None = None) -> GoldenItem:
    """Wrap a free-form prompt as a dataset item so it is sent exactly like eval prompts."""
    return GoldenItem(
        id="ask-001",
        category=Category.RAG if context else Category.FACTUAL,
        prompt=prompt.ljust(5),
        reference_answer="n/a",
        context=context or None,
        accept=["n/a"],
    )


def ask_one(
    cfg: ModelConfig,
    item: GoldenItem,
    gen: GenerationConfig,
    pricing: dict[str, PriceEntry],
    max_tokens: int | None = None,
) -> LiveAnswer:
    """Send the prompt to one model. Never raises for provider problems."""
    ok, reason = check_available(cfg)
    if not ok:
        return LiveAnswer(cfg.name, cfg.model, skipped=reason)
    request = build_request(
        item, gen.system_prompt, gen.temperature, max_tokens or gen.max_tokens, 0
    )
    try:
        result = build_provider(cfg, [item]).generate(request)
    except ProviderError as exc:
        return LiveAnswer(cfg.name, cfg.model, error=f"{exc.kind}: {str(exc)[:200]}")
    cost = compute_cost(result.input_tokens, result.output_tokens, pricing.get(cfg.model))
    return LiveAnswer(
        model_name=cfg.name,
        model_id=cfg.model,
        text=result.text.strip(),
        latency_ms=result.latency_ms,
        ttft_ms=result.ttft_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        billed_usd=cost.billed,
        list_usd=cost.list_price,
        refused=is_refusal(result.text),
        abstained=bool(item.context) and is_abstention(result.text),
    )


def ask_models(
    models: list[ModelConfig],
    prompt: str,
    gen: GenerationConfig,
    pricing: dict[str, PriceEntry],
    context: str | None = None,
    max_tokens: int | None = None,
) -> list[LiveAnswer]:
    """Ask all models in parallel; answers come back in the order of ``models``."""
    item = adhoc_item(prompt, context)
    if not models:
        return []
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        return list(pool.map(lambda m: ask_one(m, item, gen, pricing, max_tokens), models))
