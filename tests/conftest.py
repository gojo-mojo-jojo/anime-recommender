import os

# Set a dummy API key before any src modules are imported,
# so pydantic-settings doesn't fail during test collection.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
