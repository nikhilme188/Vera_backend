"""
Dataset loader — reads all seed/expanded JSON files and provides
a unified interface for accessing merchants, triggers, customers, and categories.
"""
import json
import os
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
EXPANDED_DIR = os.path.join(BASE_DIR, "expanded")


class DatasetLoader:
    """Loads and indexes all dataset entities for fast lookup."""

    def __init__(self):
        self.merchants: Dict[str, dict] = {}
        self.triggers: Dict[str, dict] = {}
        self.customers: Dict[str, dict] = {}
        self.categories: Dict[str, dict] = {}
        self.test_pairs: List[dict] = []
        self._loaded = False

    def load(self):
        """Load all dataset files."""
        if self._loaded:
            return
        self._load_categories()
        self._load_merchants()
        self._load_triggers()
        self._load_customers()
        self._load_test_pairs()
        self._loaded = True
        logger.info(
            f"Dataset loaded: {len(self.merchants)} merchants, "
            f"{len(self.triggers)} triggers, {len(self.customers)} customers, "
            f"{len(self.categories)} categories, {len(self.test_pairs)} test_pairs"
        )

    def _load_categories(self):
        cat_dir = os.path.join(DATASET_DIR, "categories")
        if not os.path.isdir(cat_dir):
            logger.warning(f"Categories directory not found: {cat_dir}")
            return
        for fname in os.listdir(cat_dir):
            if fname.endswith(".json"):
                path = os.path.join(cat_dir, fname)
                data = _load_json(path)
                if data and isinstance(data, dict):
                    slug = data.get("slug", fname.replace(".json", ""))
                    self.categories[slug] = data
        logger.info(f"Loaded {len(self.categories)} categories")

    def _load_merchants(self):
        # Try expanded first (richer data), fall back to seed
        exp_dir = os.path.join(EXPANDED_DIR, "merchants")
        if os.path.isdir(exp_dir):
            for fname in os.listdir(exp_dir):
                if fname.endswith(".json"):
                    data = _load_json(os.path.join(exp_dir, fname))
                    if data:
                        mid = data.get("merchant_id", fname.replace(".json", ""))
                        self.merchants[mid] = data

        # Seed merchants (only add if not already in expanded)
        seed_path = os.path.join(DATASET_DIR, "merchants_seed.json")
        seed = _load_json(seed_path)
        if seed and isinstance(seed, dict):
            for m in seed.get("merchants", []):
                mid = m.get("merchant_id")
                if mid and mid not in self.merchants:
                    self.merchants[mid] = m
        logger.info(f"Loaded {len(self.merchants)} merchants")

    def _load_triggers(self):
        exp_dir = os.path.join(EXPANDED_DIR, "triggers")
        if os.path.isdir(exp_dir):
            for fname in os.listdir(exp_dir):
                if fname.endswith(".json"):
                    data = _load_json(os.path.join(exp_dir, fname))
                    if data:
                        tid = data.get("id", fname.replace(".json", ""))
                        self.triggers[tid] = data

        seed_path = os.path.join(DATASET_DIR, "triggers_seed.json")
        seed = _load_json(seed_path)
        if seed and isinstance(seed, dict):
            for t in seed.get("triggers", []):
                tid = t.get("id")
                if tid and tid not in self.triggers:
                    self.triggers[tid] = t
        logger.info(f"Loaded {len(self.triggers)} triggers")

    def _load_customers(self):
        exp_dir = os.path.join(EXPANDED_DIR, "customers")
        if os.path.isdir(exp_dir):
            for fname in os.listdir(exp_dir):
                if fname.endswith(".json"):
                    data = _load_json(os.path.join(exp_dir, fname))
                    if data:
                        cid = data.get("customer_id", fname.replace(".json", ""))
                        self.customers[cid] = data

        seed_path = os.path.join(DATASET_DIR, "customers_seed.json")
        seed = _load_json(seed_path)
        if seed and isinstance(seed, dict):
            for c in seed.get("customers", []):
                cid = c.get("customer_id")
                if cid and cid not in self.customers:
                    self.customers[cid] = c
        logger.info(f"Loaded {len(self.customers)} customers")

    def _load_test_pairs(self):
        path = os.path.join(EXPANDED_DIR, "test_pairs.json")
        data = _load_json(path)
        if data and isinstance(data, dict):
            self.test_pairs = data.get("pairs", [])
        elif data and isinstance(data, list):
            self.test_pairs = data
        logger.info(f"Loaded {len(self.test_pairs)} test pairs")

    def get_merchant(self, merchant_id: str) -> Optional[dict]:
        return self.merchants.get(merchant_id)

    def get_trigger(self, trigger_id: str) -> Optional[dict]:
        return self.triggers.get(trigger_id)

    def get_customer(self, customer_id: str) -> Optional[dict]:
        return self.customers.get(customer_id)

    def get_category(self, slug: str) -> Optional[dict]:
        return self.categories.get(slug)

    def get_merchant_category(self, merchant_id: str) -> Optional[dict]:
        merchant = self.get_merchant(merchant_id)
        if merchant:
            slug = merchant.get("category_slug")
            if slug:
                return self.get_category(slug)
        return None

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Returns all dataset entities as flat documents for embedding/indexing.
        Each document has: id, type, text, metadata.
        """
        docs = []

        # Merchants
        for mid, m in self.merchants.items():
            identity = m.get("identity", {})
            perf = m.get("performance", {})
            offers = m.get("offers", [])
            signals = m.get("signals", [])
            reviews = m.get("review_themes", [])
            conv_hist = m.get("conversation_history", [])

            text_parts = [
                f"Merchant: {identity.get('name', mid)}",
                f"Category: {m.get('category_slug', 'unknown')}",
                f"City: {identity.get('city', '')}, Locality: {identity.get('locality', '')}",
                f"Owner: {identity.get('owner_first_name', '')}",
                f"Subscription: {m.get('subscription', {}).get('plan', 'N/A')} ({m.get('subscription', {}).get('status', '')})",
                f"Performance (30d): {perf.get('views', 0)} views, {perf.get('calls', 0)} calls, CTR: {perf.get('ctr', 0)}",
            ]
            if offers:
                text_parts.append(f"Active offers: {', '.join(o.get('title', '') for o in offers if o.get('status') == 'active')}")
            if signals:
                text_parts.append(f"Signals: {', '.join(signals)}")
            if reviews:
                text_parts.append(f"Review themes: {', '.join(r.get('theme', '') + '(' + r.get('sentiment', '') + ')' for r in reviews)}")
            if conv_hist:
                text_parts.append(f"Recent conversation: {conv_hist[-1].get('body', '')[:200]}")

            docs.append({
                "id": f"merchant:{mid}",
                "type": "merchant",
                "text": " | ".join(text_parts),
                "metadata": {"merchant_id": mid, "category": m.get("category_slug", "")}
            })

        # Triggers
        for tid, t in self.triggers.items():
            text = (
                f"Trigger: {tid} | Kind: {t.get('kind', '')} | Source: {t.get('source', '')} "
                f"| Scope: {t.get('scope', '')} | Urgency: {t.get('urgency', 0)} "
                f"| Merchant: {t.get('merchant_id', '')} | Payload: {json.dumps(t.get('payload', {}))}"
            )
            docs.append({
                "id": f"trigger:{tid}",
                "type": "trigger",
                "text": text,
                "metadata": {"trigger_id": tid, "kind": t.get("kind", ""), "merchant_id": t.get("merchant_id", "")}
            })

        # Customers
        for cid, c in self.customers.items():
            identity = c.get("identity", {})
            rel = c.get("relationship", {})
            text = (
                f"Customer: {identity.get('name', cid)} | Language: {identity.get('language_pref', '')} "
                f"| State: {c.get('state', '')} | Visits: {rel.get('visits_total', 0)} "
                f"| LTV: {rel.get('lifetime_value', 0)} | Services: {', '.join(rel.get('services_received', [])[:5])} "
                f"| Merchant: {c.get('merchant_id', '')}"
            )
            docs.append({
                "id": f"customer:{cid}",
                "type": "customer",
                "text": text,
                "metadata": {"customer_id": cid, "merchant_id": c.get("merchant_id", "")}
            })

        # Categories
        for slug, cat in self.categories.items():
            voice = cat.get("voice", {})
            offers = cat.get("offer_catalog", [])
            text = (
                f"Category: {cat.get('display_name', slug)} | Tone: {voice.get('tone', '')} "
                f"| Register: {voice.get('register', '')} | Vocab: {', '.join(voice.get('vocab_allowed', [])[:10])} "
                f"| Taboo: {', '.join(voice.get('vocab_taboo', [])[:5])} "
                f"| Offers: {len(offers)} available"
            )
            docs.append({
                "id": f"category:{slug}",
                "type": "category",
                "text": text,
                "metadata": {"category_slug": slug}
            })

        return docs


def _load_json(path: str) -> Optional[Any]:
    """Safely load a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


# Global singleton
dataset = DatasetLoader()
