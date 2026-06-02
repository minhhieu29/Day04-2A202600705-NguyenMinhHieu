from __future__ import annotations

import argparse
import importlib
import json
import time
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.llm import judge_answer_with_llm

ALLOWED_WEIGHT_KEYS = {"json_output", "tools", "llm_judge"}
IGNORED_EXPECTED_KEYS = {"created_at"}


def run_with_rate_limit_retry(fn, *, max_attempts: int = 6):
    attempt = 1
    while True:
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - defensive runtime wrapper
            message = str(exc).lower()
            if "rate limit" not in message and "429" not in message and "rate_limit_exceeded" not in message:
                raise
            if attempt >= max_attempts:
                raise
            sleep_seconds = min(30, 2**attempt)
            print(
                f"[retry] Rate limit detected (attempt {attempt}/{max_attempts}). "
                f"Sleeping {sleep_seconds}s before retry...",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)
            attempt += 1


@dataclass
class CaseScore:
    case_id: str
    score: float
    max_score: float
    feedback: list[str]


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    for case in cases:
        validate_case_config(case)
    return cases


def validate_case_config(case: dict[str, Any]) -> None:
    weights = case.get("weights", {})
    unknown = sorted(set(weights) - ALLOWED_WEIGHT_KEYS)
    if unknown:
        raise ValueError(f"Case {case.get('id', '<unknown>')} has unsupported weight keys: {unknown}")


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    stripped = stripped.replace("đ", "d").replace("Đ", "D")
    return " ".join(stripped.lower().split())


def coerce_result(raw: Any, *, query: str, provider: str, model_name: str | None) -> dict[str, Any]:
    if isinstance(raw, str):
        return {
            "query": query,
            "final_answer": raw,
            "tool_calls": [],
            "provider": provider,
            "model_name": model_name,
            "saved_order": None,
            "saved_order_path": None,
        }
    if isinstance(raw, dict):
        payload = dict(raw)
    elif hasattr(raw, "model_dump"):
        payload = raw.model_dump()
    elif hasattr(raw, "__dict__"):
        payload = dict(raw.__dict__)
    else:
        raise TypeError(f"Unsupported run_agent result type: {type(raw)!r}")

    payload.setdefault("query", query)
    payload.setdefault("final_answer", "")
    payload.setdefault("tool_calls", [])
    payload.setdefault("provider", provider)
    payload.setdefault("model_name", model_name)
    payload.setdefault("saved_order", None)
    payload.setdefault("saved_order_path", None)
    return payload


def load_expected_order(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT_DIR / relative_path).read_text(encoding="utf-8"))


def compare_values(expected: Any, actual: Any, *, path: str = "root") -> list[str]:
    errors: list[str] = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected object, got {type(actual).__name__}."]
        for key, value in expected.items():
            if key in IGNORED_EXPECTED_KEYS:
                continue
            if key not in actual:
                errors.append(f"{path}.{key}: missing key.")
                continue
            errors.extend(compare_values(value, actual[key], path=f"{path}.{key}"))
        return errors

    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{path}: expected list, got {type(actual).__name__}."]
        if len(expected) != len(actual):
            errors.append(f"{path}: expected {len(expected)} items, got {len(actual)}.")
            return errors

        if expected and isinstance(expected[0], dict) and "product_id" in expected[0]:
            expected = sorted(expected, key=lambda item: item["product_id"])
            actual = sorted(actual, key=lambda item: item.get("product_id", ""))

        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            errors.extend(compare_values(expected_item, actual_item, path=f"{path}[{index}]"))
        return errors

    if isinstance(expected, float):
        actual_value = float(actual)
        if abs(expected - actual_value) > 1e-9:
            errors.append(f"{path}: expected {expected}, got {actual_value}.")
        return errors

    if expected != actual:
        errors.append(f"{path}: expected {expected!r}, got {actual!r}.")
    return errors


def is_subsequence(required_tools: list[str], actual_tools: list[str]) -> bool:
    if not required_tools:
        return not actual_tools if actual_tools else True
    index = 0
    for tool_name in actual_tools:
        if tool_name == required_tools[index]:
            index += 1
            if index == len(required_tools):
                return True
    return False


def score_tools(actual_tools: list[str], required_tools: list[str], weight: float) -> tuple[float, list[str]]:
    if weight == 0:
        return 0.0, []
    if is_subsequence(required_tools, actual_tools):
        return weight, []
    return 0.0, [f"Tool trace mismatch. Expected subsequence {required_tools}, got {actual_tools}."]


def score_json_output(result: dict[str, Any], case: dict[str, Any], weight: float) -> tuple[float, list[str]]:
    if weight == 0:
        return 0.0, []

    expected = case["expected"]
    saved_order = result.get("saved_order")
    saved_order_path = result.get("saved_order_path")
    feedback: list[str] = []

    if not expected.get("expect_saved_order", False):
        if saved_order is None and not saved_order_path:
            return weight, []
        feedback.append("Order should not have been saved for this case.")
        return 0.0, feedback

    if saved_order is None:
        return 0.0, ["Missing saved_order payload."]
    if not saved_order_path:
        feedback.append("Missing saved_order_path in AgentResult.")

    expected_payload = load_expected_order(expected["expected_order_file"])
    compare_errors = compare_values(expected_payload, saved_order)
    if compare_errors:
        feedback.extend(compare_errors)
        return 0.0, feedback

    if saved_order_path:
        saved_path = Path(saved_order_path)
        if not saved_path.exists():
            feedback.append(f"Saved order file does not exist: {saved_order_path}.")
            return 0.0, feedback
        file_payload = json.loads(saved_path.read_text(encoding="utf-8"))
        file_errors = compare_values(expected_payload, file_payload, path="file")
        if file_errors:
            feedback.extend(file_errors)
            return 0.0, feedback

    return weight, feedback


def grade_result(
    result: dict[str, Any],
    case: dict[str, Any],
    *,
    judge_provider: str | None = None,
    judge_model_name: str | None = None,
) -> CaseScore:
    expected = case["expected"]
    weights = case["weights"]
    feedback: list[str] = []
    earned = 0.0

    json_score, json_feedback = score_json_output(result, case, weights.get("json_output", 0))
    tool_score, tool_feedback = score_tools(
        [tool["name"] if isinstance(tool, dict) else tool.name for tool in result.get("tool_calls", [])],
        expected.get("required_tools", []),
        weights.get("tools", 0),
    )

    earned += json_score + tool_score
    feedback.extend(json_feedback)
    feedback.extend(tool_feedback)

    judge_weight = weights.get("llm_judge", 0)
    if judge_weight and judge_provider:
        judge = judge_answer_with_llm(
            query=result["query"],
            answer=result.get("final_answer", ""),
            rubric=expected.get("grading_rubric", ""),
            provider=judge_provider,
            model_name=judge_model_name,
        )
        earned += judge_weight * (judge["score"] / 10)
        feedback.extend(judge["feedback"])

    return CaseScore(
        case_id=case["id"],
        score=round(earned, 2),
        max_score=float(sum(weights.get(key, 0) for key in ALLOWED_WEIGHT_KEYS)),
        feedback=feedback,
    )


def summarize_scores(scores: list[CaseScore]) -> dict[str, Any]:
    total_earned = sum(item.score for item in scores)
    total_max = sum(item.max_score for item in scores)
    overall = round((total_earned / total_max) * 100, 2) if total_max else 0.0
    return {
        "overall_score": overall,
        "total_earned": total_earned,
        "total_max": total_max,
        "cases": [
            {
                "case_id": item.case_id,
                "score": item.score,
                "max_score": item.max_score,
                "feedback": item.feedback,
            }
            for item in scores
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Grade saved JSON output for the order-agent lab")
    parser.add_argument("--module", default="solution.agent.graph")
    parser.add_argument("--cases", default=str(ROOT_DIR / "data" / "graded_cases.json"))
    parser.add_argument("--provider", default="google", choices=["google", "openai", "custom", "ollama"])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--today", default="2026-06-01")
    parser.add_argument("--pass-threshold", type=float, default=80.0)
    parser.add_argument("--judge-provider", default=None, choices=["google", "openai", "custom", "ollama"])
    parser.add_argument("--judge-model-name", default=None)
    args = parser.parse_args()

    module = importlib.import_module(args.module)
    if not hasattr(module, "run_agent"):
        raise SystemExit(f"Module {args.module} does not expose run_agent()")

    cases = load_cases(Path(args.cases))
    effective_judge_provider = args.judge_provider
    if effective_judge_provider is None and any(case["weights"].get("llm_judge", 0) > 0 for case in cases):
        effective_judge_provider = args.provider

    scores: list[CaseScore] = []
    for case in cases:
        raw_result = run_with_rate_limit_retry(
            lambda: module.run_agent(
                case["query"],
                provider=args.provider,
                model_name=args.model_name,
                today=args.today,
            )
        )
        result = coerce_result(raw_result, query=case["query"], provider=args.provider, model_name=args.model_name)
        scores.append(
            run_with_rate_limit_retry(
                lambda: grade_result(
                    result,
                    case,
                    judge_provider=effective_judge_provider,
                    judge_model_name=args.judge_model_name,
                )
            )
        )

    summary = summarize_scores(scores)
    try:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    except UnicodeEncodeError:
        # Windows terminals may use cp1252 and fail on symbols like "₫".
        print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0 if summary["overall_score"] >= args.pass_threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
