"""
Configuration for the magicpin AI Challenge bot.
Uses Groq LLM provider.
"""
import os
from dotenv import load_dotenv

# Load .env from the same directory as this config file
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

# ======================== EDIT THIS ========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# ===========================================================

BOT_PORT = int(os.environ.get("BOT_PORT", "8080"))
BOT_HOST = os.environ.get("BOT_HOST", "0.0.0.0")

# Team metadata
TEAM_NAME = "Team Nikhil"
TEAM_MEMBERS = ["Nikhil"]
CONTACT_EMAIL = "nikhil@example.com"
BOT_VERSION = "1.0.0"

# Composition settings — temperature=0.0 for deterministic output
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 1500
LLM_TIMEOUT_SECONDS = 25  # Leave 5s headroom against 30s judge budget

# Tick settings
MAX_ACTIONS_PER_TICK = 20
