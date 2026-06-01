"""Financial disclaimer injection for FinSage-7B responses.

Every public answer carries a mandatory disclaimer. These helpers append it
cleanly and avoid duplicating it when the model has already echoed it back.
"""

from __future__ import annotations

#: Mandatory disclaimer appended to every model response.
FINANCIAL_DISCLAIMER = (
    "FinSage-7B is not a licensed financial advisor. Outputs are not investment "
    "recommendations. Always verify responses against the original filing."
)


def get_disclaimer() -> str:
    """Return the canonical financial disclaimer.

    Returns:
        The :data:`FINANCIAL_DISCLAIMER` string.
    """
    return FINANCIAL_DISCLAIMER


def remove_duplicate_disclaimer(text: str) -> str:
    """Strip any occurrences of the disclaimer from ``text``.

    Useful before re-appending the disclaimer so it appears exactly once even
    if the model echoed it into its own output.

    Args:
        text: Text that may contain zero or more copies of the disclaimer.

    Returns:
        ``text`` with every embedded disclaimer removed and surrounding
        whitespace collapsed.
    """
    cleaned = text.replace(FINANCIAL_DISCLAIMER, "")
    # Collapse blank lines left behind by removal, then trim.
    lines = [line.rstrip() for line in cleaned.splitlines()]
    return "\n".join(lines).strip()


def append_disclaimer(answer: str) -> str:
    """Append the disclaimer to ``answer`` exactly once.

    If the answer already ends with (or contains) the disclaimer it is first
    removed so the result never repeats it.

    Args:
        answer: The generated answer text.

    Returns:
        The answer followed by a blank line and the disclaimer.
    """
    base = remove_duplicate_disclaimer(answer)
    if not base:
        return FINANCIAL_DISCLAIMER
    return f"{base}\n\n{FINANCIAL_DISCLAIMER}"
