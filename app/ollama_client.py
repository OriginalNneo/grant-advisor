"""
Thin wrapper around Ollama's tool-calling chat loop.

The model is allowed to call the functions defined in tools.py. We keep
calling ollama.chat(), executing whatever tool_calls come back and feeding
the results in as 'tool' messages, until the model returns a plain answer
(or we hit a safety cap on iterations).
"""
from __future__ import annotations

import json
import os

from . import tools

CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "gemma4:e2b")
MAX_TOOL_ITERATIONS = int(os.environ.get("OLLAMA_MAX_TOOL_ITERATIONS", "8"))

# Gemma 4 ships with a default context window of only 4K tokens regardless of
# the model's actual capacity, which is too small once a PDF's extracted text
# plus a multi-turn tool-calling conversation are in play. Override it. Safe
# to bump higher (e.g. 65536) if you have the RAM/VRAM and use bigger PDFs.
NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "32768"))

try:
    import ollama
    _OLLAMA_IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover - missing package, or a broken/incompatible install
    ollama = None
    _OLLAMA_IMPORT_ERROR = e


class OllamaUnavailable(RuntimeError):
    """Raised when the ollama package is missing or the local Ollama server can't be reached."""


def _normalize_arguments(raw_args) -> dict:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return {}
    return {}


def run_agent_loop(
    messages: list[dict],
    model: str | None = None,
    max_iterations: int = MAX_TOOL_ITERATIONS,
) -> tuple[str, list[dict]]:
    """Run the tool-calling loop. Returns (final_text, full_conversation)."""
    if ollama is None:
        detail = f" Underlying error: {_OLLAMA_IMPORT_ERROR}" if _OLLAMA_IMPORT_ERROR else ""
        raise OllamaUnavailable(
            "The 'ollama' Python package isn't available (either not installed, or it failed "
            f"to import). Run: pip install ollama{detail}"
        )

    model = model or CHAT_MODEL
    conversation: list[dict] = list(messages)

    for _ in range(max_iterations):
        try:
            response = ollama.chat(
                model=model,
                messages=conversation,
                tools=tools.TOOL_SCHEMAS,
                options={"num_ctx": NUM_CTX},
            )
        except Exception as e:
            raise OllamaUnavailable(
                f"Couldn't reach Ollama with model '{model}'. Make sure Ollama is running "
                f"(`ollama serve`) and the model is pulled (`ollama pull {model}`). "
                f"Original error: {e}"
            ) from e

        message = response["message"]
        # Normalize to a plain dict so it serializes cleanly in the conversation history.
        message_dict = {
            "role": message.get("role", "assistant"),
            "content": message.get("content", ""),
        }
        tool_calls = message.get("tool_calls")
        if tool_calls:
            message_dict["tool_calls"] = tool_calls
        conversation.append(message_dict)

        if not tool_calls:
            return message_dict["content"], conversation

        for call in tool_calls:
            fn = call.get("function", {})
            fn_name = fn.get("name", "")
            fn_args = _normalize_arguments(fn.get("arguments"))
            result = tools.call_tool(fn_name, fn_args)
            conversation.append({
                "role": "tool",
                "name": fn_name,
                "content": result,
            })

    return (
        "(Stopped after reaching the max tool-call iteration limit without a final answer. "
        "Try asking a narrower follow-up question.)",
        conversation,
    )


def chat_once(prompt: str, model: str | None = None, num_ctx: int | None = None) -> str:
    """Single-shot, unconstrained chat call: no tools, no JSON schema, just
    prompt -> plain text answer. Used for freeform Q&A (app/grant_qa.py)
    where forcing structured JSON output isn't needed or wanted -- unlike
    extract_json, there's no schema to retry against, so this makes exactly
    one call.
    """
    if ollama is None:
        detail = f" Underlying error: {_OLLAMA_IMPORT_ERROR}" if _OLLAMA_IMPORT_ERROR else ""
        raise OllamaUnavailable(
            "The 'ollama' Python package isn't available (either not installed, or it failed "
            f"to import). Run: pip install ollama{detail}"
        )

    model = model or CHAT_MODEL
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_ctx": num_ctx or NUM_CTX, "temperature": 0.2},
        )
    except Exception as e:
        raise OllamaUnavailable(
            f"Couldn't reach Ollama with model '{model}'. Make sure Ollama is running "
            f"(`ollama serve`) and the model is pulled (`ollama pull {model}`). "
            f"Original error: {e}"
        ) from e

    return response["message"].get("content", "").strip()


_RETRY_PROMPT_SUFFIX = (
    "\n\nIMPORTANT: Your previous response could not be parsed as valid JSON. Respond with ONLY "
    "the JSON object matching the required schema -- no markdown code fences, no commentary, no "
    "text before or after the JSON."
)


def _try_parse_json(content: str) -> dict | None:
    """Parse `content` as JSON, tolerating minor formatting noise (e.g. a stray
    markdown code fence or leading/trailing prose) by falling back to the
    outermost {...} substring before giving up.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def extract_json(prompt: str, schema: dict, model: str | None = None, num_ctx: int | None = None) -> dict:
    """Single-shot structured extraction: no tools, no loop, just prompt -> JSON.

    Passes the JSON schema straight to Ollama's `format` parameter, which
    constrains the model's output grammar (supported by recent Ollama/gemma
    tool-calling models). Used for one-off document parsing, not agentic
    reasoning -- keep that on run_agent_loop.

    `num_ctx` overrides the module-level default (NUM_CTX) for this call only
    -- useful for callers feeding in unusually long documents (e.g. full
    multi-page proposal text) that need a bigger context window than the
    default 32768 without changing it globally via OLLAMA_NUM_CTX.

    If the model's response can't be parsed as JSON, retries once with a
    shorter, more insistent prompt (schema-format constraints are usually
    respected, but small local models occasionally wrap output in prose or a
    code fence anyway). Raises ValueError only if both attempts fail.
    """
    if ollama is None:
        detail = f" Underlying error: {_OLLAMA_IMPORT_ERROR}" if _OLLAMA_IMPORT_ERROR else ""
        raise OllamaUnavailable(
            "The 'ollama' Python package isn't available (either not installed, or it failed "
            f"to import). Run: pip install ollama{detail}"
        )

    model = model or CHAT_MODEL
    active_prompt = prompt
    content = ""
    for attempt in range(2):
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": active_prompt}],
                format=schema,
                options={"num_ctx": num_ctx or NUM_CTX, "temperature": 0.0},
            )
        except Exception as e:
            raise OllamaUnavailable(
                f"Couldn't reach Ollama with model '{model}'. Make sure Ollama is running "
                f"(`ollama serve`) and the model is pulled (`ollama pull {model}`). "
                f"Original error: {e}"
            ) from e

        content = response["message"].get("content", "")
        parsed = _try_parse_json(content)
        if parsed is not None:
            return parsed
        active_prompt = prompt + _RETRY_PROMPT_SUFFIX

    raise ValueError(f"Model did not return valid JSON after retry.\nRaw content: {content[:500]}")
