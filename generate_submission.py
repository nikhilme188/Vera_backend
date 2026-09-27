"""
Generate submission.jsonl from the 30 canonical test pairs.
Reads test_pairs.json + expanded dataset, composes each message via the LLM,
and writes one JSON line per test pair.

Usage: python generate_submission.py
"""
import json
import sys
import os
import time
import asyncio
from pathlib import Path

# Imports from local composer package (same directory)

from composer.llm_client import call_llm
from composer.prompts import build_composition_prompt
from composer.validate import validate_composition

# Paths
EXPANDED_DIR = Path(__file__).parent / "expanded"
TEST_PAIRS_PATH = EXPANDED_DIR / "test_pairs.json"
OUTPUT_PATH = Path(__file__).parent / "submission.jsonl"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def main():
    # Load test pairs
    test_pairs = load_json(TEST_PAIRS_PATH)["pairs"]
    print(f"Loaded {len(test_pairs)} test pairs")

    # Load all expanded data
    categories = {}
    for f in (EXPANDED_DIR / "categories").glob("*.json"):
        data = load_json(f)
        categories[data["slug"]] = data

    merchants = {}
    for f in (EXPANDED_DIR / "merchants").glob("*.json"):
        data = load_json(f)
        merchants[data["merchant_id"]] = data

    customers = {}
    for f in (EXPANDED_DIR / "customers").glob("*.json"):
        data = load_json(f)
        customers[data["customer_id"]] = data

    triggers = {}
    for f in (EXPANDED_DIR / "triggers").glob("*.json"):
        data = load_json(f)
        triggers[data["id"]] = data

    print(f"Loaded: {len(categories)} categories, {len(merchants)} merchants, "
          f"{len(customers)} customers, {len(triggers)} triggers")

    results = []

    for i, pair in enumerate(test_pairs):
        test_id = pair["test_id"]
        trigger_id = pair["trigger_id"]
        merchant_id = pair["merchant_id"]
        customer_id = pair.get("customer_id")

        print(f"\n[{i+1}/{len(test_pairs)}] {test_id}: {trigger_id}")

        trigger = triggers.get(trigger_id)
        merchant = merchants.get(merchant_id)
        if not trigger or not merchant:
            print(f"  SKIP: trigger={bool(trigger)}, merchant={bool(merchant)}")
            results.append({
                "test_id": test_id,
                "body": "",
                "cta": "none",
                "send_as": "vera",
                "suppression_key": trigger_id,
                "rationale": "Context missing",
            })
            continue

        category_slug = merchant.get("category_slug", trigger.get("payload", {}).get("category", ""))
        category = categories.get(category_slug, {})
        customer = customers.get(customer_id) if customer_id else None

        # Build prompts
        system_prompt, user_prompt = build_composition_prompt(
            category=category,
            merchant=merchant,
            trigger=trigger,
            customer=customer,
        )

        # Call LLM
        result = await call_llm(system_prompt, user_prompt)
        if not result:
            print(f"  FAIL: LLM returned None")
            results.append({
                "test_id": test_id,
                "body": "",
                "cta": "none",
                "send_as": "vera",
                "suppression_key": trigger.get("suppression_key", trigger_id),
                "rationale": "LLM failure",
            })
            continue

        # Validate
        is_valid, issues = validate_composition(result, category, merchant, trigger, customer, [])
        if issues:
            print(f"  ISSUES: {issues}")
        if not is_valid:
            print(f"  INVALID — retrying...")
            fix_prompt = user_prompt + f"\n\nPREVIOUS ATTEMPT HAD ISSUES: {', '.join(issues)}\nFix and compose again."
            result = await call_llm(system_prompt, fix_prompt)
            if not result:
                results.append({
                    "test_id": test_id,
                    "body": "",
                    "cta": "none",
                    "send_as": "vera",
                    "suppression_key": trigger.get("suppression_key", trigger_id),
                    "rationale": "Validation failure",
                })
                continue

        entry = {
            "test_id": test_id,
            "body": result.get("body", ""),
            "cta": result.get("cta", "none"),
            "send_as": result.get("send_as", "vera"),
            "suppression_key": result.get("suppression_key", trigger.get("suppression_key", trigger_id)),
            "rationale": result.get("rationale", ""),
        }
        results.append(entry)
        print(f"  OK: {entry['body'][:80]}...")

        # Rate-limit to avoid API throttling
        time.sleep(1)

    # Write JSONL
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n✅ Written {len(results)} lines to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
