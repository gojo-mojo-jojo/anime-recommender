import os

# Set dummy API keys before any src modules are imported,
# so pydantic-settings doesn't fail during test collection.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
