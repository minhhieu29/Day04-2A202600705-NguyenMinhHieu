from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def normalize_content(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        text = raw.get("text")
        return str(text).strip() if text is not None else str(raw).strip()
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            text = normalize_content(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(raw).strip()


def build_chat_model(
    *,
    provider: str = "google",
    model_name: str | None = None,
    temperature: float = 0.0,
):
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name or os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        base_url = os.getenv("OPENAI_API_BASE") or os.getenv("LLM_ENDPOINT")
        return ChatOpenAI(
            model=model_name or os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )
    if provider == "custom":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name or os.getenv("CUSTOM_LLM_MODEL"),
            openai_api_key=os.getenv("CUSTOM_LLM_KEY"),
            openai_api_base=os.getenv("CUSTOM_LLM_URL"),
            temperature=temperature,
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_name or os.getenv("OLLAMA_MODEL", "qwen3.5:3b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )
    raise ValueError("This lab supports only the `google`, `openai`, `custom`, and `ollama` providers.")


def extract_json_object(raw: Any) -> dict[str, Any]:
    text = normalize_content(raw)
    if "```" in text:
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if blocks:
            text = blocks[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start : end + 1])


def judge_answer_with_llm(
    *,
    query: str,
    answer: str,
    rubric: str,
    provider: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    prompt = f"""
You are grading a student order-agent answer.
Return JSON only with:
- score: integer from 0 to 10
- verdict: short string
- feedback: short list of strings

Rubric:
{rubric}

User query:
{query}

Student answer:
{answer}
""".strip()
    payload = extract_json_object(model.invoke(prompt).content)
    score = max(0, min(10, int(payload.get("score", 0))))
    return {
        "score": score,
        "verdict": str(payload.get("verdict", "")).strip(),
        "feedback": [str(item).strip() for item in payload.get("feedback", []) if str(item).strip()],
    }
