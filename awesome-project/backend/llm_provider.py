"""
Provider-agnostic LLM interface.

Pipeline nodes call `complete_json(step, prompt)` and never know which model
or provider is behind it. To swap Gemini for Groq, Ollama, or Claude later,
change GeminiProvider (or add a new provider class) and `get_provider()` --
nothing in backend/pipeline/ needs to change.

Uses the current `google-genai` SDK (the older `google.generativeai` package
is fully deprecated as of 2026 -- no more updates or bug fixes).
"""
from abc import ABC, abstractmethod
import json
from google import genai
from google.genai import types
from backend import config


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, system: str = None, json_mode: bool = False) -> str:
        ...


class GeminiProvider(LLMProvider):
    _client = None  # shared across instances -- one client, many models

    def __init__(self, model_name: str):
        self.model_name = model_name
        if GeminiProvider._client is None:
            GeminiProvider._client = genai.Client(api_key=config.GEMINI_API_KEY)

    def complete(self, prompt: str, system: str = None, json_mode: bool = False) -> str:
        gen_config = types.GenerateContentConfig(
            system_instruction=system if system else None,
            response_mime_type="application/json" if json_mode else None,
        )
        response = GeminiProvider._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=gen_config,
        )
        return response.text


def get_provider(step: str) -> LLMProvider:
    """Look up which model a pipeline step should use, per config.MODEL_ROUTING."""
    model_name = config.MODEL_ROUTING.get(step, "gemini-2.5-flash-lite")
    return GeminiProvider(model_name)


def _strip_fences(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1] if cleaned.count("```") >= 2 else cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    return cleaned


def complete_json(step: str, prompt: str, system: str = None) -> dict:
    """Call the provider routed to `step` and parse JSON output.

    Includes one retry-with-repair pass since small/fast models occasionally
    wrap JSON in markdown fences or add stray text despite instructions.
    """
    provider = get_provider(step)
    raw = provider.complete(prompt, system=system, json_mode=True)
    try:
        return json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        repair_prompt = (
            "The following was supposed to be valid JSON but failed to parse. "
            f"Return ONLY corrected valid JSON, nothing else:\n\n{raw}"
        )
        raw2 = provider.complete(repair_prompt, json_mode=True)
        return json.loads(_strip_fences(raw2))
