"""
Conversation handlers for /v1/reply endpoint.
Handles:
  1. Auto-reply detection — identical merchant messages = canned responses
  2. Intent-transition detection — merchant commits, bot switches to action mode
  3. Hostile/stop detection — merchant wants to end, bot exits gracefully
  4. Off-topic redirection — politely redirect to mission
  5. Graceful exit — after too many unanswered nudges

These handlers run BEFORE the LLM to decide:
  - If we should skip the LLM entirely (auto-reply, end conditions)
  - If we should augment the LLM prompt (intent transition detected)
"""
import re
import logging
from typing import Optional
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


# ============================================================
# AUTO-REPLY DETECTION
# ============================================================

AUTO_REPLY_PATTERNS = [
    "thank you for contacting",
    "our team will respond shortly",
    "our team will get back to you",
    "we have received your message",
    "thank you for reaching out",
    "we will respond at the earliest",
    "please wait while we connect you",
    "this is an automated message",
    "auto-reply",
    "automatic reply",
    "we'll be right with you",
    "thanks for your message",
    "thank you for your message",
    "we are currently unavailable",
    "office hours",
    "business hours",
]


def is_auto_reply(message: str) -> bool:
    """
    Detect if a message is a canned WhatsApp Business auto-reply.
    Uses keyword matching against known patterns.
    """
    msg_lower = message.lower().strip()

    # Check against known patterns
    for pattern in AUTO_REPLY_PATTERNS:
        if pattern in msg_lower:
            return True

    return False


def is_near_identical(msg1: str, msg2: str, threshold: float = 0.85) -> bool:
    """Check if two messages are near-identical using sequence matching."""
    return SequenceMatcher(None, msg1.lower().strip(), msg2.lower().strip()).ratio() >= threshold


def count_repeated_messages(messages: list[str], current_message: str) -> int:
    """Count how many times a near-identical message has appeared (including current)."""
    count = 0
    for msg in messages:
        if is_near_identical(msg, current_message):
            count += 1
    return count + 1  # +1 for the current message


# ============================================================
# INTENT TRANSITION DETECTION
# ============================================================

COMMITMENT_PHRASES_EN = [
    "let's do it",
    "lets do it",
    "go ahead",
    "proceed",
    "ok proceed",
    "ok lets do",
    "ok let's do",
    "yes do it",
    "yes please",
    "yes, do it",
    "yes go ahead",
    "sure go ahead",
    "sounds good",
    "ok sounds good",
    "what's next",
    "whats next",
    "what next",
    "do it",
    "i want to join",
    "sign me up",
    "count me in",
    "i'm in",
    "im in",
    "let's go",
    "lets go",
    "start it",
    "yes start",
    "confirm",
    "i confirm",
    "agreed",
    "ok done",
    "done deal",
    "ok i am interested",
    "ok i'm interested",
    "yes interested",
    "ok activate",
]

COMMITMENT_PHRASES_HI = [
    "haan chalo",
    "haan chaliye",
    "kar do",
    "kar dena",
    "karo",
    "kar dijiye",
    "theek hai",
    "thik hai",
    "bilkul",
    "zarur",
    "aage badho",
    "aage badhiye",
    "haan ji",
    "haan",
    "acha chalo",
    "sahi hai",
    "done",
    "chalu karo",
    "shuru karo",
    "ho jayega",
]


def detect_intent_transition(message: str) -> bool:
    """
    Detect if the merchant is committing / saying yes.
    Returns True if the merchant is ready to proceed.
    """
    msg_lower = message.lower().strip()

    # Check exact phrases
    for phrase in COMMITMENT_PHRASES_EN + COMMITMENT_PHRASES_HI:
        if phrase in msg_lower:
            return True

    # Short affirmative messages (< 20 chars) with yes/ok/sure
    if len(msg_lower) < 30:
        short_affirms = ["yes", "ok", "sure", "yep", "yeah", "yup", "ya", "haan", "ji"]
        words = msg_lower.split()
        if len(words) <= 5 and any(w in short_affirms for w in words):
            return True

    return False


# ============================================================
# HOSTILITY / STOP DETECTION
# ============================================================

STOP_PHRASES = [
    "stop",
    "stop messaging",
    "unsubscribe",
    "not interested",
    "don't message",
    "dont message",
    "don't contact",
    "dont contact",
    "leave me alone",
    "spam",
    "this is spam",
    "useless",
    "stop sending",
    "remove me",
    "take me off",
    "never contact",
    "block",
    "band karo",
    "mat bhejo",
    "mat karo",
    "nahi chahiye",
    "koi zarurat nahi",
    "bakwaas",
]

HOSTILE_PHRASES = [
    "why are you bothering",
    "this is useless",
    "waste of time",
    "who asked you",
    "get lost",
    "shut up",
    "fraud",
    "scam",
    "cheating",
    "bakwaas band karo",
    "paisa barbaad",
    "bekar",
    "chutiya",
    "bevkuf",
]


def detect_stop(message: str) -> bool:
    """Detect if merchant explicitly wants to stop."""
    msg_lower = message.lower().strip()
    for phrase in STOP_PHRASES:
        if phrase in msg_lower:
            return True
    return False


def detect_hostility(message: str) -> bool:
    """Detect if merchant is being hostile/aggressive."""
    msg_lower = message.lower().strip()
    for phrase in HOSTILE_PHRASES:
        if phrase in msg_lower:
            return True
    return False


# ============================================================
# MAIN HANDLER: respond()
# ============================================================

def respond(
    conversation_id: str,
    merchant_id: str,
    message: str,
    turn_number: int,
    merchant_messages: list[str],
    sent_bodies: list[str],
    merchant: dict,
    category: dict,
) -> Optional[dict]:
    """
    Pre-LLM conversation handler.
    Returns a response dict if the handler can decide without LLM,
    or None if the LLM should be consulted.

    Response format:
        {"action": "send", "body": "...", "cta": "...", "rationale": "..."}
        {"action": "wait", "wait_seconds": N, "rationale": "..."}
        {"action": "end", "rationale": "..."}
    """
    msg_lower = message.lower().strip()

    # --- 1. STOP / HOSTILE DETECTION (highest priority) ---
    if detect_stop(message):
        logger.info(f"[{conversation_id}] STOP detected: '{message[:50]}'")
        return {
            "action": "end",
            "rationale": "Merchant explicitly opted out. Closing conversation; suppressing future messages."
        }

    if detect_hostility(message):
        logger.info(f"[{conversation_id}] HOSTILITY detected: '{message[:50]}'")
        # Check language preference for apology
        languages = merchant.get("identity", {}).get("languages", ["en"])
        if "hi" in languages:
            body = "Maafi chahta hoon — aage se message nahi bhejunga. Agar kabhi zarurat ho toh 'Hi Vera' likh dijiyega. 🙏"
        else:
            body = "Apologies — I won't message again. If anything changes, you can always restart with 'Hi Vera'. 🙏"

        return {
            "action": "send",
            "body": body,
            "cta": "none",
            "rationale": "Merchant hostility detected. One-line apology + opt-out path; conversation will close after."
        }

    # --- 2. AUTO-REPLY DETECTION ---
    if is_auto_reply(message):
        # Count ALL previous merchant messages that are near-identical (including non-auto-reply ones)
        auto_count = 0
        for prev_msg in merchant_messages:
            if is_auto_reply(prev_msg) and is_near_identical(prev_msg, message):
                auto_count += 1
        auto_count += 1  # +1 for current message
        
        logger.info(f"[{conversation_id}] Auto-reply #{auto_count}: '{message[:50]}'")

        if auto_count >= 3:
            # 3+ auto-replies: END
            return {
                "action": "end",
                "rationale": f"Auto-reply detected {auto_count}x in a row, no real engagement. Closing conversation."
            }
        elif auto_count >= 2:
            # 2nd auto-reply: WAIT long
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "Same auto-reply twice in a row — owner not at phone. Wait 24h before retry."
            }
        else:
            # 1st auto-reply: try once more
            owner_name = merchant.get("identity", {}).get("owner_first_name", "")
            languages = merchant.get("identity", {}).get("languages", ["en"])
            if "hi" in languages and owner_name:
                body = f"Lagta hai ye auto-reply hai 😊 {owner_name} ji jab dekhein, bas 'Yes' bol dijiye."
            elif owner_name:
                body = f"Looks like an auto-reply 😊 When {owner_name} sees this, just reply 'Yes' to continue."
            else:
                body = "Looks like an auto-reply 😊 When the owner sees this, just reply 'Yes' to continue."

            return {
                "action": "send",
                "body": body,
                "cta": "binary_yes_no",
                "rationale": "Detected auto-reply (canned phrasing). One explicit prompt to flag it for the owner."
            }

    # --- 3. Near-identical repeated messages (non-auto-reply patterns) ---
    repeat_count = count_repeated_messages(merchant_messages, message)
    if repeat_count >= 3:
        logger.info(f"[{conversation_id}] Repeated message #{repeat_count}: '{message[:50]}'")
        return {
            "action": "end",
            "rationale": f"Merchant sent near-identical message {repeat_count}x — likely auto-reply or not engaging. Ending conversation."
        }

    # --- 4. INTENT TRANSITION ---
    if detect_intent_transition(message):
        logger.info(f"[{conversation_id}] INTENT TRANSITION: '{message[:50]}'")
        # Handle directly — don't delegate to LLM which may hesitate
        owner_name = merchant.get("identity", {}).get("owner_first_name", "")
        languages = merchant.get("identity", {}).get("languages", ["en"])
        name_part = f" {owner_name} ji" if owner_name else ""
        
        if "hi" in languages:
            body = f"Badiya{name_part}! Main aapke liye ek plan draft kar rahi hoon — 2 min mein bhejti hoon. Bas confirm kar dijiye tab."
        else:
            body = f"Great{name_part}! I'm drafting a plan for you right now — will send it in 2 minutes. Just confirm when ready."
        
        return {
            "action": "send",
            "body": body,
            "cta": "binary_confirm_cancel",
            "rationale": "Intent transition detected — merchant committed. Switching to action mode immediately with concrete next step."
        }

    # --- 5. Let the LLM handle it ---
    return None
