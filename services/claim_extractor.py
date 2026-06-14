"""Claim extraction orchestration."""

from __future__ import annotations

from services.llm_service import LLMService
from utils.constants import CLAIM_TYPES
from utils.helpers import clean_claim_text, remove_duplicate_claims


class ClaimExtractor:
    """Extract, clean, and deduplicate factual claims."""

    def __init__(self, model: str | None = None):
        self.llm_service = LLMService(model=model)

    def extract_and_process_claims(
        self,
        text: str,
        remove_duplicates: bool = True,
        max_claims: int = 100,
    ) -> tuple[list[dict[str, str]], str | None]:
        """Extract factual claims from long document text."""
        if not text or len(text.strip()) < 100:
            return [], "Text is too short to extract meaningful claims."

        all_claims: list[dict[str, str]] = []
        errors: list[str] = []

        for chunk in self._chunk_text(text):
            claims, error = self.llm_service.extract_claims(chunk)
            if error:
                errors.append(error)
            all_claims.extend(claims)

        cleaned = self._validate_and_clean_claims(all_claims)
        if remove_duplicates:
            cleaned = remove_duplicate_claims(cleaned)

        if not cleaned and errors:
            return [], "; ".join(sorted(set(errors)))

        return cleaned[:max_claims], None

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 10000, overlap: int = 600) -> list[str]:
        """Split text into overlapping character chunks."""
        chunks: list[str] = []
        start = 0
        text = text.strip()

        while start < len(text) and len(chunks) < 8:
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            if end == len(text):
                break
            start = max(0, end - overlap)

        return chunks

    @staticmethod
    def _validate_and_clean_claims(claims: list[dict[str, str]]) -> list[dict[str, str]]:
        cleaned: list[dict[str, str]] = []
        for claim_data in claims:
            claim = clean_claim_text(str(claim_data.get("claim", "")))
            claim_type = str(claim_data.get("type", "Other")).strip()

            if len(claim) < 12:
                continue
            if len(claim) > 600:
                claim = claim[:597].rstrip() + "..."
            if claim_type not in CLAIM_TYPES:
                claim_type = "Other"

            cleaned.append({"claim": claim, "type": claim_type})

        return cleaned
