"""
Groq API client for message composition using LangChain.
Configured for deterministic output (temperature=0, seed=42).
"""
import json
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from config import GROQ_API_KEY, GROQ_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


def get_llm(temperature: float = None, max_tokens: int = None) -> ChatGroq | None:
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY is not set! Set it in .env or via environment variable.")
        return None

    temp = temperature if temperature is not None else LLM_TEMPERATURE
    tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS

    try:
        return ChatGroq(
            model=GROQ_MODEL,
            temperature=temp,
            max_tokens=tokens,
            timeout=LLM_TIMEOUT_SECONDS,
            api_key=GROQ_API_KEY,
            # Determinism: seed ensures reproducible output
            model_kwargs={"seed": 42},
        )
    except Exception as e:
        logger.error(f"Failed to initialize ChatGroq: {e}")
        return None


async def call_llm(system_prompt: str, user_prompt: str, temperature: float = None,
             max_tokens: int = None, response_format: str = "json_object"):
    """
    Call Groq chat completions API via LangChain asynchronously.
    If response_format='text', returns raw string.
    If response_format='json_object', returns parsed JSON dict.
    Returns None on failure. Output is deterministic (temperature=0, seed=42).
    """
    llm = get_llm(temperature, max_tokens)
    if not llm:
        return None

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    try:
        response = await llm.ainvoke(messages)
        content = str(response.content)

        # Return raw text if requested
        if response_format == "text":
            return content

        # Parse JSON from response
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            import re
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                return json.loads(match.group())
            logger.error(f"Could not parse JSON from LLM response: {content[:200]}")
            return None

    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        return None


async def call_llm_text(system_prompt: str, user_prompt: str, temperature: float = None,
                  max_tokens: int = None) -> str | None:
    """
    Call Groq API and return raw text (no JSON parsing) via LangChain asynchronously.
    Output is deterministic for the same input.
    """
    llm = get_llm(temperature, max_tokens)
    if not llm:
        return None

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    try:
        response = await llm.ainvoke(messages)
        return str(response.content)
    except Exception as e:
        logger.error(f"Groq text call failed: {e}")
        return None
