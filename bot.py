
import json
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import (
    BOT_PORT, BOT_HOST, TEAM_NAME, TEAM_MEMBERS, CONTACT_EMAIL,
    BOT_VERSION, GROQ_MODEL, MAX_ACTIONS_PER_TICK,
)
from state import ContextStore, ConversationState, SentSuppression
from composer.llm_client import call_llm
from composer.prompts import build_composition_prompt, build_reply_prompt
from composer.validate import validate_composition, quick_fabrication_check
from conversation_handlers import respond as handler_respond
from dataset_loader import dataset
from rag_engine import rag_engine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bot")

app = FastAPI(title="magicpin Bot", version=BOT_VERSION)
START_TIME = time.time()

ctx_store = ContextStore()
conv_state = ConversationState()
suppression = SentSuppression()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize dataset and RAG engine on server start."""
    logger.info("Loading dataset...")
    dataset.load()

    # Pre-populate context store from dataset
    for slug, cat in dataset.categories.items():
        ctx_store.upsert("category", slug, 1, cat)
    for mid, m in dataset.merchants.items():
        ctx_store.upsert("merchant", mid, 1, m)
    for tid, t in dataset.triggers.items():
        ctx_store.upsert("trigger", tid, 1, t)
    for cid, c in dataset.customers.items():
        ctx_store.upsert("customer", cid, 1, c)
    logger.info("Context store pre-populated from dataset")

    # Initialize RAG (embedding + Pinecone indexing)
    docs = dataset.get_all_documents()
    rag_engine.initialize(docs)
    logger.info(f"RAG engine ready: {rag_engine.is_ready}")

# Root endpoint (Frontend is deployed separately)
@app.get("/")
async def dashboard():
    return {"status": "Vera API is running. Frontend is hosted separately."}

# API: get all loaded contexts for dashboard
@app.get("/v1/contexts")
async def get_contexts():
    return {
        "categories": {k: v["payload"] for k, v in ctx_store._store["category"].items()},
        "merchants": {k: v["payload"] for k, v in ctx_store._store["merchant"].items()},
        "triggers": {k: v["payload"] for k, v in ctx_store._store["trigger"].items()},
        "customers": {k: v["payload"] for k, v in ctx_store._store["customer"].items()},
    }

# API: get all conversation states for dashboard
@app.get("/v1/conversations")
async def get_conversations():
    with conv_state._lock:
        return dict(conv_state._conversations)



def resolve_contexts(trigger_id: str) -> tuple[dict, dict, dict, dict | None]:
    """
    Given a trigger_id, resolve the full (category, merchant, trigger, customer) tuple.
    Returns (category, merchant, trigger, customer_or_None).
    """
    trigger = ctx_store.get("trigger", trigger_id)
    if not trigger:
        return {}, {}, {}, None

    merchant_id = trigger.get("merchant_id", "")
    customer_id = trigger.get("customer_id")
    merchant = ctx_store.get("merchant", merchant_id) or {}
    category_slug = merchant.get("category_slug", trigger.get("payload", {}).get("category", ""))
    category = ctx_store.get("category", category_slug) or {}
    customer = ctx_store.get("customer", customer_id) if customer_id else None

    return category, merchant, trigger, customer


async def compose_message(category: dict, merchant: dict, trigger: dict,
                    customer: dict | None, conv_id: str = None) -> dict | None:
    """
    Compose a message using the LLM.
    Returns the composed message dict or None on failure.
    """
    # Get conversation history for anti-repetition
    sent_bodies = conv_state.get_sent_bodies(conv_id) if conv_id else []

    # Retrieve RAG context
    rag_context = ""
    if rag_engine.is_ready:
        merchant_id = merchant.get("merchant_id", "")
        category_slug = merchant.get("category_slug", "")
        rag_context = rag_engine.get_context_for_conversation(
            merchant_id=merchant_id,
            message=f"{trigger.get('kind', '')} {trigger.get('id', '')}",
            category_slug=category_slug,
        )

    # Build prompts
    system_prompt, user_prompt = build_composition_prompt(
        category=category,
        merchant=merchant,
        trigger=trigger,
        customer=customer,
        conversation_history=merchant.get("conversation_history"),
    )

    # Inject RAG context into user prompt
    if rag_context:
        user_prompt = f"=== RETRIEVED CONTEXT (RAG) ===\n{rag_context}\n\n{user_prompt}"

    # Call LLM
    result = await call_llm(system_prompt, user_prompt)
    if not result:
        logger.error("LLM returned None for composition")
        return None

    # Validate
    is_valid, issues = validate_composition(
        result=result,
        category=category,
        merchant=merchant,
        trigger=trigger,
        customer=customer,
        sent_bodies=sent_bodies,
    )

    if issues:
        logger.warning(f"Composition issues: {issues}")

    if not is_valid:
        # One retry with explicit fix instructions
        logger.info("Retrying composition with fix instructions...")
        fix_prompt = user_prompt + f"\n\nPREVIOUS ATTEMPT HAD ISSUES: {', '.join(issues)}\nFix these issues and compose again."
        result = await call_llm(system_prompt, fix_prompt)
        if not result:
            return None

        # Validate again
        is_valid2, issues2 = validate_composition(result, category, merchant, trigger, customer, sent_bodies)
        if not is_valid2:
            logger.error(f"Retry also failed validation: {issues2}")
            return None

    # Quick fabrication check
    fab_issues = quick_fabrication_check(
        result.get("body", ""), category, merchant, trigger, customer
    )
    if fab_issues:
        logger.warning(f"Fabrication check: {fab_issues}")

    return result


# ============================================================
# ENDPOINT 1: GET /v1/healthz
# ============================================================

@app.get("/v1/healthz")
async def healthz():
    counts = ctx_store.counts()
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
    }


# ============================================================
# ENDPOINT 2: GET /v1/metadata
# ============================================================

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": TEAM_NAME,
        "team_members": TEAM_MEMBERS,
        "model": GROQ_MODEL,
        "approach": "Groq-powered single-prompt composer with per-trigger-kind framing dispatch, "
                    "auto-reply detection, intent-transition handling, and post-LLM validation",
        "contact_email": CONTACT_EMAIL,
        "version": BOT_VERSION,
        "submitted_at": datetime.now(timezone.utc).isoformat() + "Z",
    }


# ============================================================
# ENDPOINT 3: POST /v1/context
# ============================================================

@app.post("/v1/context")
async def push_context(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_json", "details": "Could not parse JSON body"},
        )

    scope = body.get("scope")
    context_id = body.get("context_id")
    version = body.get("version")
    payload = body.get("payload")

    if not scope or not context_id or version is None or payload is None:
        return JSONResponse(
            status_code=400,
            content={
                "accepted": False,
                "reason": "missing_fields",
                "details": "Required: scope, context_id, version, payload",
            },
        )

    if scope not in ("category", "merchant", "customer", "trigger"):
        return JSONResponse(
            status_code=400,
            content={
                "accepted": False,
                "reason": "invalid_scope",
                "details": f"scope must be one of: category, merchant, customer, trigger. Got: {scope}",
            },
        )

    result = ctx_store.upsert(scope, context_id, version, payload)

    if result["accepted"]:
        logger.info(f"Context stored: {scope}/{context_id} v{version}")
        return JSONResponse(status_code=200, content=result)
    elif result.get("reason") == "stale_version":
        return JSONResponse(status_code=409, content=result)
    else:
        return JSONResponse(status_code=400, content=result)


# ============================================================
# ENDPOINT 4: POST /v1/tick
# ============================================================

@app.post("/v1/tick")
async def tick(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"actions": []})

    now = body.get("now", datetime.now(timezone.utc).isoformat())
    available_triggers = body.get("available_triggers", [])

    actions = []

    # Track which merchants we've already sent to this tick
    merchants_sent = set()

    for trigger_id in available_triggers:
        if len(actions) >= MAX_ACTIONS_PER_TICK:
            break

        # Check suppression
        trigger = ctx_store.get("trigger", trigger_id)
        if not trigger:
            logger.warning(f"Trigger {trigger_id} not found in store, skipping")
            continue

        sup_key = trigger.get("suppression_key", trigger_id)
        if suppression.is_suppressed(sup_key):
            logger.info(f"Trigger {trigger_id} suppressed (key: {sup_key})")
            continue

        merchant_id = trigger.get("merchant_id", "")

        # One action per merchant per tick
        if merchant_id in merchants_sent:
            continue

        # Check if conversation already ended for this merchant
        active_convs = conv_state.get_active_conversations_for_merchant(merchant_id)
        # Don't spam a merchant with multiple open conversations
        if len(active_convs) >= 2:
            continue

        # Check trigger expiry
        expires = trigger.get("expires_at")
        if expires:
            try:
                exp_dt = datetime.fromisoformat(expires.replace("Z", "+00:00"))
                if datetime.now(timezone.utc) > exp_dt:
                    logger.info(f"Trigger {trigger_id} expired, skipping")
                    continue
            except Exception:
                pass

        # Resolve all contexts
        category, merchant, trigger_data, customer = resolve_contexts(trigger_id)

        if not merchant:
            logger.warning(f"Merchant {merchant_id} not found for trigger {trigger_id}")
            continue

        # Generate conversation ID
        customer_id = trigger.get("customer_id")
        conv_id = f"conv_{merchant_id}_{trigger_id}"

        # Compose the message
        try:
            result = await compose_message(category, merchant, trigger_data, customer, conv_id)
        except Exception as e:
            logger.error(f"Composition failed for {trigger_id}: {e}")
            continue

        if not result:
            continue

        # Build the action
        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": result.get("send_as", "vera"),
            "trigger_id": trigger_id,
            "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
            "template_params": [],
            "body": result.get("body", ""),
            "cta": result.get("cta", "none"),
            "suppression_key": result.get("suppression_key", sup_key),
            "rationale": result.get("rationale", ""),
        }

        # Record state
        conv_state.record_sent(conv_id, action["body"], merchant_id, trigger_id)
        suppression.suppress(sup_key)
        merchants_sent.add(merchant_id)
        actions.append(action)

        logger.info(f"Action composed: {trigger_id} -> {merchant_id} ({len(action['body'])} chars)")

    return {"actions": actions}


# ============================================================
# ENDPOINT 5: POST /v1/reply
# ============================================================

@app.post("/v1/reply")
async def reply(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"action": "end", "rationale": "Could not parse JSON body"},
        )

    conv_id = body.get("conversation_id", "")
    merchant_id = body.get("merchant_id", "")
    customer_id = body.get("customer_id")
    from_role = body.get("from_role", "merchant")
    message = body.get("message", "")
    turn_number = body.get("turn_number", 1)

    # Check if conversation already ended
    if conv_state.is_ended(conv_id):
        return {"action": "end", "rationale": "Conversation was previously ended."}

    # Record the merchant message
    conv_state.record_merchant_message(conv_id, message)

    # Get merchant and category context
    merchant = ctx_store.get("merchant", merchant_id) or {}
    category_slug = merchant.get("category_slug", "")
    category = ctx_store.get("category", category_slug) or {}

    # Get conversation state
    state = conv_state.get_state(conv_id)
    merchant_messages = state.get("merchant_messages", [])
    sent_bodies = state.get("sent_bodies", [])

    # --- Pre-LLM handler ---
    handler_result = handler_respond(
        conversation_id=conv_id,
        merchant_id=merchant_id,
        message=message,
        turn_number=turn_number,
        merchant_messages=merchant_messages[:-1],  # Exclude current (already added above)
        sent_bodies=sent_bodies,
        merchant=merchant,
        category=category,
    )

    if handler_result:
        # Handler decided — record and return
        if handler_result.get("action") == "end":
            conv_state.mark_ended(conv_id)
        elif handler_result.get("action") == "send":
            conv_state.record_sent(conv_id, handler_result.get("body", ""), merchant_id)

        logger.info(f"Handler decided for {conv_id}: {handler_result.get('action')}")
        return handler_result

    # --- LLM-powered reply ---
    # Find the trigger associated with this conversation
    trigger_id = state.get("trigger_id")
    trigger = ctx_store.get("trigger", trigger_id) if trigger_id else None
    customer = ctx_store.get("customer", customer_id) if customer_id else None

    system_prompt, user_prompt = build_reply_prompt(
        category=category,
        merchant=merchant,
        trigger=trigger,
        customer=customer,
        merchant_message=message,
        conversation_state=state,
        conversation_id=conv_id,
    )

    try:
        result = await call_llm(system_prompt, user_prompt)
    except Exception as e:
        logger.error(f"LLM reply failed for {conv_id}: {e}")
        return {"action": "wait", "wait_seconds": 1800, "rationale": "Internal error; backing off."}

    if not result:
        return {"action": "wait", "wait_seconds": 1800, "rationale": "LLM unavailable; backing off."}

    action = result.get("action", "wait")

    # Validate
    if action == "send":
        reply_body = result.get("body", "")
        if not reply_body:
            return {"action": "wait", "wait_seconds": 1800, "rationale": "Empty reply body; backing off."}

        # Anti-repetition check
        if conv_state.is_body_repeated(conv_id, reply_body):
            logger.warning(f"Reply body is repeated in {conv_id}, modifying...")
            # Try to get a different response
            fix_prompt = user_prompt + "\n\nIMPORTANT: Your previous response was identical to an already-sent message. Write a DIFFERENT response."
            result2 = await call_llm(system_prompt, fix_prompt)
            if result2 and result2.get("action") == "send" and result2.get("body"):
                result = result2
            else:
                return {"action": "wait", "wait_seconds": 3600, "rationale": "Could not generate non-repeated reply."}

        conv_state.record_sent(conv_id, result.get("body", ""), merchant_id)

    elif action == "end":
        conv_state.mark_ended(conv_id)

    response = {
        "action": result.get("action", "wait"),
    }

    if action == "send":
        response["body"] = result.get("body", "")
        response["cta"] = result.get("cta", "none")
    elif action == "wait":
        response["wait_seconds"] = result.get("wait_seconds", 1800)

    response["rationale"] = result.get("rationale", "")

    logger.info(f"Reply for {conv_id}: action={action}")
    return response


# ============================================================
# ENDPOINT: POST /v1/start — Bot-first greeting
# ============================================================

@app.post("/v1/start")
async def start_conversation(request: Request):
    """
    Bot sends the first message to greet the merchant.
    Frontend calls this when the chat page loads.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    conv_id = body.get("conversation_id", f"chat_{uuid.uuid4().hex[:8]}")
    merchant_id = body.get("merchant_id", "001")

    # Look up merchant from dataset
    merchant = dataset.get_merchant(merchant_id)
    if not merchant:
        # Use first available merchant as fallback
        if dataset.merchants:
            merchant_id = list(dataset.merchants.keys())[0]
            merchant = dataset.merchants[merchant_id]
        else:
            merchant = {}

    identity = merchant.get("identity", {})
    name = identity.get("owner_first_name", "there")
    biz_name = identity.get("name", "your business")
    category_slug = merchant.get("category_slug", "")
    category = dataset.get_category(category_slug) or {}

    # Build greeting using RAG context
    rag_context = ""
    if rag_engine.is_ready:
        rag_context = rag_engine.get_context_for_conversation(
            merchant_id=merchant_id,
            message=f"greeting introduction for {biz_name}",
            category_slug=category_slug,
        )

    # Use LLM to generate a personalized greeting
    greeting_prompt = (
        f"You are Vera, magicpin's AI growth assistant. Generate a warm, personalized "
        f"first message to greet the merchant.\n"
        f"Merchant: {biz_name} (Owner: {name})\n"
        f"Category: {category.get('display_name', category_slug)}\n"
        f"City: {identity.get('city', '')}\n"
    )

    if rag_context:
        greeting_prompt += f"\nRelevant context:\n{rag_context[:500]}\n"

    greeting_prompt += (
        f"\nRules:\n"
        f"- Keep it under 2 sentences\n"
        f"- Be warm and specific to their business\n"
        f"- Mention one actionable thing you can help with\n"
        f"- Respond with ONLY the greeting text, no JSON\n"
    )

    greeting = await call_llm(
        system_prompt="You are Vera, magicpin's friendly AI assistant. Reply with only the greeting text.",
        user_prompt=greeting_prompt,
        response_format="text",
    )

    if isinstance(greeting, dict):
        greeting_text = greeting.get("body", greeting.get("message", str(greeting)))
    elif isinstance(greeting, str):
        greeting_text = greeting
    else:
        greeting_text = f"Hi {name}! I'm Vera, your magicpin AI assistant for {biz_name}. How can I help you grow today?"

    # Initialize conversation state
    conv_state.record_sent(conv_id, greeting_text, merchant_id)

    return {
        "action": "send",
        "body": greeting_text,
        "conversation_id": conv_id,
        "merchant_id": merchant_id,
        "merchant_name": biz_name,
    }


# ============================================================
# OPTIONAL: POST /v1/teardown
# ============================================================

@app.post("/v1/teardown")
async def teardown():
    ctx_store.clear()
    conv_state.clear()
    suppression.clear()
    logger.info("Teardown complete — all state wiped")
    return {"status": "torn_down"}


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    logger.info(f"Starting bot on {BOT_HOST}:{BOT_PORT}")
    uvicorn.run(app, host=BOT_HOST, port=BOT_PORT, log_level="info")
