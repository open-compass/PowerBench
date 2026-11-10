from __future__ import annotations


class EligibilityRejection(ValueError):
    """A valid corpus item that is unsuitable for a particular QA template."""

    def __init__(self, reason: str, detail: str) -> None:
        if not reason or not reason.replace("_", "").isalnum():
            raise ValueError(f"Invalid eligibility-rejection reason: {reason!r}")
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")