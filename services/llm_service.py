"""OpenAI-powered LLM operations."""

from __future__ import annotations

import os
from typing import Any

from openai import APIError, OpenAI, RateLimitError

from utils.constants import (
    CLAIM_EXTRACTION_SYSTEM_PROMPT,
    CLAIM_EXTRACTION_TEMPERATURE,
    CLAIM_EXTRACTION_USER_PROMPT,
    DEFAULT_MODEL,
    ERROR_API_KEY,
    LLM_TEMPERATURE,
    VALID_STATUSES,
    VERIFICATION_SYSTEM_PROMPT,
    VERIFICATION_USER_PROMPT,
    STATUS_UNVERIFIABLE,
)
from utils.helpers import extract_json_from_text, safe_int


class LLMService:
    """Thin service wrapper around OpenAI Chat Completions."""

    def __init__(self, model: str | None = None):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(ERROR_API_KEY)

        self.client = OpenAI(api_key=api_key, timeout=60, max_retries=2)
        self.model = model or DEFAULT_MODEL

    def extract_claims(self, text: str) -> tuple[list[dict[str, str]], str | None]:
        """Extract factual claims as structured JSON."""
        try:
            response_text = self._json_chat(
                system_prompt=CLAIM_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=CLAIM_EXTRACTION_USER_PROMPT.format(text=text[:12000]),
                temperature=CLAIM_EXTRACTION_TEMPERATURE,
                max_tokens=2500,
            )
            payload = extract_json_from_text(response_text, {"claims": []})
            claims = payload.get("claims", payload if isinstance(payload, list) else [])
            if not isinstance(claims, list):
                return [], "OpenAI returned an invalid claims payload."

            return [
                {"claim": str(item.get("claim", "")), "type": str(item.get("type", "Other"))}
                for item in claims
                if isinstance(item, dict) and item.get("claim")
            ], None
        except RateLimitError:
            return [], "OpenAI rate limit reached. Please try again later."
        except APIError as exc:
            return [], f"OpenAI API error: {exc}"
        except Exception as exc:
            return [], f"Claim extraction error: {exc}"

    def verify_claim(self, claim: str, evidence: str) -> tuple[dict[str, Any], str | None]:
        """Verify one claim against web evidence."""
        try:
            response_text = self._json_chat(
                system_prompt=VERIFICATION_SYSTEM_PROMPT,
                user_prompt=VERIFICATION_USER_PROMPT.format(
                    claim=claim,
                    evidence=evidence[:9000],
                ),
                temperature=LLM_TEMPERATURE,
                max_tokens=900,
            )
            payload = extract_json_from_text(response_text, {})
            if not isinstance(payload, dict):
                return self.default_verification("Could not parse verification JSON."), None

            status = str(payload.get("status", STATUS_UNVERIFIABLE)).strip()
            if status not in VALID_STATUSES:
                status = STATUS_UNVERIFIABLE

            return {
                "status": status,
                "confidence": safe_int(payload.get("confidence"), default=0),
                "explanation": str(payload.get("explanation", "")).strip(),
                "key_finding": str(payload.get("key_finding", "")).strip(),
            }, None
        except RateLimitError:
            return self.default_verification("OpenAI rate limit reached."), None
        except APIError as exc:
            return self.default_verification(f"OpenAI API error: {exc}"), None
        except Exception as exc:
            return self.default_verification(f"Verification error: {exc}"), None

    def _json_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call Chat Completions in JSON-object mode."""
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or "{}"

    @staticmethod
    def default_verification(explanation: str) -> dict[str, Any]:
        return {
            "status": STATUS_UNVERIFIABLE,
            "confidence": 0,
            "explanation": explanation,
            "key_finding": "",
        }
