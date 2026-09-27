"""
Post-LLM validation layer.
Checks every composed message BEFORE returning it.
Catches: multiple CTAs, fabricated data, anti-repetition, empty bodies, taboo words, URLs.
Re-prompts the LLM once on validation failure (soft retry).
"""
import re
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def validate_composition(
    result: dict,
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None,
    sent_bodies: list[str],
) -> tuple[bool, list[str]]:
    """
    Validate a composed message.
    Returns: (is_valid, list_of_issues).
    """
    issues = []

    if not result:
        return False, ["LLM returned None"]

    body = result.get("body", "")
    cta = result.get("cta", "")
    send_as = result.get("send_as", "")
    suppression_key = result.get("suppression_key", "")
    rationale = result.get("rationale", "")

    # ====== HARD FAILURES (must re-prompt) ======

    # 1. Empty body
    if not body or not body.strip():
        issues.append("EMPTY_BODY: Message body is empty")
        return False, issues

    # 2. Missing required fields
    for field in ["body", "cta", "send_as", "suppression_key", "rationale"]:
        if not result.get(field):
            issues.append(f"MISSING_FIELD: {field} is missing or empty")

    # 3. Anti-repetition: exact match against sent bodies
    if body in sent_bodies:
        issues.append("REPETITION: Body is verbatim identical to a previously sent message")
        return False, issues

    # 4. URL detection (Meta rejects)
    url_pattern = r'https?://[^\s]+'
    if re.search(url_pattern, body):
        issues.append("URL_FOUND: Body contains a URL — Meta will reject this")
        return False, issues

    # ====== SOFT WARNINGS (deduct points but don't re-prompt) ======

    # 5. Check for taboo words
    taboos = category.get("voice", {}).get("vocab_taboo", [])
    body_lower = body.lower()
    for taboo in taboos:
        if taboo.lower() in body_lower:
            issues.append(f"TABOO_WORD: Body contains taboo word '{taboo}'")

    # 6. Check for multiple CTAs (more than one question mark at the end)
    # Simple heuristic: count sentences ending with "?"
    questions = [s.strip() for s in re.split(r'[.!]', body) if s.strip().endswith('?')]
    if len(questions) > 2:
        issues.append(f"MULTIPLE_CTAS: Body has {len(questions)} question-like sentences — keep to 1 CTA")

    # 7. Check send_as matches customer presence
    if customer and send_as != "merchant_on_behalf":
        issues.append("SEND_AS_MISMATCH: Customer context present but send_as is not 'merchant_on_behalf'")

    if not customer and send_as == "merchant_on_behalf":
        issues.append("SEND_AS_MISMATCH: No customer context but send_as is 'merchant_on_behalf'")

    # 8. Check for long preambles
    preamble_patterns = [
        r'^(hi|hello|hey),?\s*(i\s+hope|how\s+are\s+you|greetings|good\s+(morning|afternoon|evening))',
        r'^(dear|respected)\s+',
    ]
    for pattern in preamble_patterns:
        if re.search(pattern, body_lower):
            issues.append("PREAMBLE: Body starts with a long preamble")
            break

    # 9. Check body length — warn if too long for WhatsApp
    if len(body) > 1000:
        issues.append(f"LONG_BODY: Body is {len(body)} chars — consider trimming for WhatsApp")

    # 10. Language check: if merchant has "hi" in languages, body should have some Hindi
    languages = merchant.get("identity", {}).get("languages", [])
    if "hi" in languages:
        # Check for at least some Devanagari or common Hindi-English words
        hindi_markers = ["aap", "kya", "hai", "hain", "ke", "ka", "ki", "ko", "se",
                         "mein", "apne", "ye", "wo", "ek", "hum", "aur", "par",
                         "liye", "apka", "apki", "bhi", "toh", "nahi", "ji",
                         "chahiye", "kaise", "kab", "kahan", "kyun",
                         "haan", "zarur", "shukriya", "dhanyavaad"]
        has_hindi = any(marker in body_lower.split() for marker in hindi_markers)
        # Also check for Devanagari script
        has_devanagari = bool(re.search(r'[\u0900-\u097F]', body))
        if not has_hindi and not has_devanagari:
            issues.append("LANGUAGE_MISS: Merchant prefers hi-en but body appears to be pure English")

    has_hard_failure = any(issue.startswith(("EMPTY_BODY", "REPETITION", "URL_FOUND")) for issue in issues)
    return not has_hard_failure, issues


def quick_fabrication_check(body: str, category: dict, merchant: dict,
                            trigger: dict, customer: dict | None) -> list[str]:
    """
    Quick check for obviously fabricated data.
    Looks for numbers in the body and checks if they appear in any context.
    This is a lightweight heuristic — not a full entity check.
    """
    issues = []

    # Extract all numbers from body (3+ digits to avoid false positives)
    body_numbers = set(re.findall(r'\b\d{3,}\b', body))

    if not body_numbers:
        return issues

    # Build a set of all numbers from the contexts
    context_text = (
        json.dumps(category) +
        json.dumps(merchant) +
        json.dumps(trigger) +
        (json.dumps(customer) if customer else "")
    )
    context_numbers = set(re.findall(r'\b\d{3,}\b', context_text))

    # Numbers in body but not in any context
    fabricated = body_numbers - context_numbers
    if fabricated:
        # Allow some common numbers (years, common prices)
        common_numbers = {"100", "200", "500", "1000", "2000", "2024", "2025", "2026", "2027"}
        truly_fabricated = fabricated - common_numbers
        if truly_fabricated:
            issues.append(f"POSSIBLE_FABRICATION: Numbers {truly_fabricated} appear in body but not in any context")

    return issues
