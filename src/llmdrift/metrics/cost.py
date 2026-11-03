"""Per-query cost from token counts and list prices."""

from __future__ import annotations

from dataclasses import dataclass

from llmdrift.config import PriceEntry


@dataclass(frozen=True)
class Cost:
    """Two views of cost for one query, in USD.

    ``billed`` is what you actually pay (0 on a free tier or local model).
    ``list_price`` is what the same tokens would cost at the paid list price
    (``None`` when no list price exists, e.g. a local model, or tokens are unknown).
    """

    billed: float
    list_price: float | None


def compute_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    price: PriceEntry | None,
    using_free_tier: bool = True,
) -> Cost:
    """Cost of one query. Unknown model or tokens give ``list_price=None``."""
    if (
        price is None
        or input_tokens is None
        or output_tokens is None
        or price.input_per_1m is None
        or price.output_per_1m is None
    ):
        return Cost(billed=0.0, list_price=None)
    list_price = (input_tokens * price.input_per_1m + output_tokens * price.output_per_1m) / 1e6
    billed = 0.0 if (using_free_tier and price.free_tier) else list_price
    return Cost(billed=billed, list_price=list_price)
