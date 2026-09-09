"""Lightweight helpers for resolving sharded-evaluation media descriptors."""

from __future__ import annotations

from typing import Any


def resolve_lazy_hf_media(
    value: Any,
    marker: str,
    default_column: str,
) -> tuple[bool, Any]:
    """Resolve one HF Dataset row descriptor without importing model code."""
    if not isinstance(value, dict) or not value.get(marker):
        return False, value
    dataset = value.get("dataset")
    row_index = value.get("row_index")
    column = value.get("column", default_column)
    if dataset is None or not isinstance(row_index, int):
        raise ValueError(f"Invalid lazy Hugging Face media descriptor: {value}")
    try:
        return True, dataset[row_index][column]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Failed to resolve lazy media row {row_index}, column {column!r}"
        ) from exc
