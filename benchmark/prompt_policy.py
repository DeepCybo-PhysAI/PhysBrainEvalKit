from typing import Final


PROMPT_POLICIES: Final = ("original",)
ORIGINAL_PROMPT_SOURCE_COMMIT: Final = (
    "5ec3a60e03894ea5b9215127b47afaf39d969dae"
)


def validate_prompt_policy(value: str) -> str:
    if value not in PROMPT_POLICIES:
        choices = ", ".join(PROMPT_POLICIES)
        raise ValueError(f"Unsupported prompt policy {value!r}; choose from {choices}")
    return value
