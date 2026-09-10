"""Helpers for resolving project datasets from the Hugging Face Hub cache."""

from pathlib import Path


def resolve_snapshot(dataset_name: str) -> Path:
    """Return a local snapshot for a Hub dataset ID or an existing path."""
    path = Path(dataset_name).expanduser()
    if path.exists():
        return path.resolve()
    if path.is_absolute() or dataset_name.startswith(".") or len(path.parts) != 2:
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=dataset_name, repo_type="dataset"))
