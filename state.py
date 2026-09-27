"""
In-memory context store + conversation state.
Thread-safe via threading locks.
Stores: categories, merchants, customers, triggers.
Tracks conversation state for anti-repetition and auto-reply detection.
"""
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Literal


class ContextStore:
    """
    Versioned in-memory store for all four context scopes.
    Key = (scope, context_id), value = {version, payload, stored_at}.
    Idempotent: same version = no-op (409), higher version = atomic replace, lower = reject.
    """

    def __init__(self):
        self._lock = threading.Lock()
        
        self._store: Dict[str, Dict[str, Dict[str, Any]]] = {
            "category": {},
            "merchant": {},
            "customer": {},
            "trigger": {},
        }

    def upsert(self, scope: str, context_id: str, version: int, payload: dict) -> dict:
        """
        Idempotent upsert.
        Returns: {accepted, ack_id, stored_at} on success, or
                 {accepted:false, reason, current_version} on conflict/error.
        """
        if scope not in self._store:
            return {
                "accepted": False,
                "reason": "invalid_scope",
                "details": f"scope must be one of: category, merchant, customer, trigger. Got: {scope}",
            }

        with self._lock:
            bucket = self._store[scope]
            existing = bucket.get(context_id)

            if existing is not None:
                if version == existing["version"]:
                    
                    return {"accepted": True, "ack_id": f"ack_{context_id}_v{version}", "stored_at": existing["stored_at"]}
                elif version < existing["version"]:
                    return {
                        "accepted": False,
                        "reason": "stale_version",
                        "current_version": existing["version"],
                    }

            now = datetime.now(timezone.utc).isoformat()
            bucket[context_id] = {
                "version": version,
                "payload": payload,
                "stored_at": now,
            }
            ack_id = f"ack_{context_id}_v{version}"
            return {"accepted": True, "ack_id": ack_id, "stored_at": now}

    def get(self, scope: str, context_id: str) -> Optional[dict]:
        """Get the current payload for a given scope+context_id."""
        with self._lock:
            entry = self._store.get(scope, {}).get(context_id)
            return entry["payload"] if entry else None

    def get_all(self, scope: str) -> Dict[str, dict]:
        """Get all payloads in a scope. Returns {context_id: payload}."""
        with self._lock:
            return {
                cid: entry["payload"]
                for cid, entry in self._store.get(scope, {}).items()
            }

    def count(self, scope: str) -> int:
        """Count entries in a scope."""
        with self._lock:
            return len(self._store.get(scope, {}))

    def counts(self) -> Dict[str, int]:
        """Return counts for all scopes."""
        with self._lock:
            return {scope: len(bucket) for scope, bucket in self._store.items()}

    def clear(self):
        """Wipe everything (for teardown)."""
        with self._lock:
            for scope in self._store:
                self._store[scope].clear()


class ConversationState:
    """
    Tracks per-conversation state:
    - sent_bodies: list of body strings (for anti-repetition)
    - merchant_messages: list of merchant messages (for auto-reply detection)
    - turn_count: how many turns have occurred
    - ended: whether the conversation has been ended
    - last_action_at: timestamp of last action
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._conversations: Dict[str, Dict[str, Any]] = {}

    def _ensure(self, conv_id: str):
        if conv_id not in self._conversations:
            self._conversations[conv_id] = {
                "sent_bodies": [],
                "merchant_messages": [],
                "turn_count": 0,
                "ended": False,
                "last_action_at": None,
                "merchant_id": None,
                "trigger_id": None,
            }

    def record_sent(self, conv_id: str, body: str, merchant_id: str = None, trigger_id: str = None):
        """Record a body we sent in this conversation."""
        with self._lock:
            self._ensure(conv_id)
            self._conversations[conv_id]["sent_bodies"].append(body)
            self._conversations[conv_id]["turn_count"] += 1
            self._conversations[conv_id]["last_action_at"] = datetime.now(timezone.utc).isoformat()
            if merchant_id:
                self._conversations[conv_id]["merchant_id"] = merchant_id
            if trigger_id:
                self._conversations[conv_id]["trigger_id"] = trigger_id

    def record_merchant_message(self, conv_id: str, message: str):
        """Record a merchant message for auto-reply detection."""
        with self._lock:
            self._ensure(conv_id)
            self._conversations[conv_id]["merchant_messages"].append(message)
            self._conversations[conv_id]["turn_count"] += 1

    def is_body_repeated(self, conv_id: str, body: str) -> bool:
        """Check if a body has already been sent in this conversation."""
        with self._lock:
            self._ensure(conv_id)
            return body in self._conversations[conv_id]["sent_bodies"]

    def get_sent_bodies(self, conv_id: str) -> list:
        """Get all sent bodies for a conversation."""
        with self._lock:
            self._ensure(conv_id)
            return list(self._conversations[conv_id]["sent_bodies"])

    def get_merchant_messages(self, conv_id: str) -> list:
        """Get all merchant messages for a conversation."""
        with self._lock:
            self._ensure(conv_id)
            return list(self._conversations[conv_id]["merchant_messages"])

    def mark_ended(self, conv_id: str):
        """Mark a conversation as ended."""
        with self._lock:
            self._ensure(conv_id)
            self._conversations[conv_id]["ended"] = True

    def is_ended(self, conv_id: str) -> bool:
        """Check if a conversation is ended."""
        with self._lock:
            return self._conversations.get(conv_id, {}).get("ended", False)

    def get_state(self, conv_id: str) -> dict:
        """Get full conversation state."""
        with self._lock:
            self._ensure(conv_id)
            return dict(self._conversations[conv_id])

    def get_active_conversations_for_merchant(self, merchant_id: str) -> list:
        """List active (not ended) conversation ids for a merchant."""
        with self._lock:
            return [
                cid
                for cid, state in self._conversations.items()
                if state.get("merchant_id") == merchant_id and not state.get("ended")
            ]

    def clear(self):
        """Wipe all conversation state."""
        with self._lock:
            self._conversations.clear()


class SentSuppression:
    """
    Tracks suppression keys to avoid sending the same trigger/message type
    multiple times within a window.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._keys: set = set()

    def is_suppressed(self, key: str) -> bool:
        with self._lock:
            return key in self._keys

    def suppress(self, key: str):
        with self._lock:
            self._keys.add(key)

    def clear(self):
        with self._lock:
            self._keys.clear()
