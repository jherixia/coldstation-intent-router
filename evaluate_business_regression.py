"""Run the 20-question cold-station business regression after capability expansion."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from router import ALL_INTENTS, LayeredIntentRouter


BASE_DIR = Path(__file__).resolve().parent
CASES_PATH = BASE_DIR / "business_scope_eval_20_regression.csv"
BASELINE_PATH = BASE_DIR / "business_scope_eval_20_results.csv"
RESULTS_PATH = BASE_DIR / "business_scope_eval_20_results_v2.csv"
SUMMARY_PATH = BASE_DIR / "business_scope_eval_20_summary_v2.json"


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def handling_matches(expected: str, actual: str, response: str) -> bool:
    if actual != expected:
        return False
    if expected == "internal.explain":
        return "机制" in response and "知识库" in response
    if expected == "internal.savings_estimate":
        return "仅供参考" in response and "不生成估算数值" in response
    if expected == "internal.capability_missing":
        return "接口后续接入中" in response
    if expected == "reject_control":
        return "不会执行" in response
    return True


def main() -> None:
    with CASES_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        cases = list(csv.DictReader(source))
    with BASELINE_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        baseline = {row["id"]: row for row in csv.DictReader(source)}
    if len(cases) != 20 or set(row["id"] for row in cases) != set(baseline):
        raise ValueError("regression set must preserve the original 20 IDs")
    if any(row["expected_result"] not in ALL_INTENTS for row in cases):
        raise ValueError("regression set contains an unsupported expected result")

    router = LayeredIntentRouter()
    records: list[dict[str, Any]] = []
    for case in cases:
        result = router.route(case["text"])
        l1 = next((step for step in result["trace"] if step["layer"] == "L1"), None)
        l2 = next((step for step in result["trace"] if step["layer"] == "L2"), None)
        before = baseline[case["id"]]
        correct = handling_matches(
            case["expected_result"],
            result["final_intent"],
            result["simulated_response"],
        )
        records.append({
            **case,
            "before_intent": before["actual_intent"],
            "before_layer": before["hit_layer"],
            "before_correct": before["correct"],
            "actual_intent": result["final_intent"],
            "hit_layer": result["hit_layer"],
            "qwen_called": result["called_small_llm"],
            "total_seconds": result["total_seconds"],
            "correct": correct,
            "simulated_response": result["simulated_response"],
            "control_triggered": False,
            "l1_candidate": l1.get("candidate", "") if l1 else "",
            "l1_similarity": l1.get("similarity", "") if l1 else "",
            "l1_margin": l1.get("margin", "") if l1 else "",
            "l1_accepted": l1.get("accepted", "") if l1 else "",
            "l2_status": l2.get("status", "") if l2 else "",
            "l2_intent": l2.get("intent", "") if l2 else "",
            "l2_elapsed_seconds": l2.get("elapsed_seconds", "") if l2 else "",
            "fallback_reason": result["fallback_reason"] or "",
        })

    with RESULTS_PATH.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    durations = [float(row["total_seconds"]) for row in records]
    layer_counts = Counter(row["hit_layer"] for row in records)
    old_supported = [row for row in records if baseline[row["id"]]["support_status"] == "SUPPORTED"]
    old_correct_ids = {
        row["id"] for row in old_supported if baseline[row["id"]]["correct"] == "True"
    }
    old_error_ids = {
        row["id"] for row in old_supported if baseline[row["id"]]["correct"] == "False"
    }
    old_missing_ids = {
        row["id"] for row in records
        if baseline[row["id"]]["support_status"] == "NOT_IMPLEMENTED"
    }

    summary = {
        "test_set": {
            "path": str(CASES_PATH),
            "sha256": hashlib.sha256(CASES_PATH.read_bytes()).hexdigest(),
            "total": len(records),
        },
        "before": {
            "supported": 13,
            "correct": 9,
            "not_implemented": 7,
        },
        "after": {
            "executed": len(records),
            "correct": sum(bool(row["correct"]) for row in records),
            "accuracy": sum(bool(row["correct"]) for row in records) / len(records),
            "layer_hits": dict(layer_counts),
            "qwen_calls": sum(bool(row["qwen_called"]) for row in records),
            "latency_seconds": {
                "average": statistics.fmean(durations),
                "p50": percentile(durations, 0.50),
                "p95": percentile(durations, 0.95),
            },
            "control_trigger_count": sum(bool(row["control_triggered"]) for row in records),
            "errors": [
                {
                    "id": row["id"],
                    "text": row["text"],
                    "expected_result": row["expected_result"],
                    "actual_intent": row["actual_intent"],
                    "hit_layer": row["hit_layer"],
                }
                for row in records if not row["correct"]
            ],
        },
        "regression": {
            "fixed_original_errors": sorted(
                row["id"] for row in records if row["id"] in old_error_ids and row["correct"]
            ),
            "regressed_original_correct": sorted(
                row["id"] for row in records if row["id"] in old_correct_ids and not row["correct"]
            ),
            "implemented_previous_missing": sorted(
                row["id"] for row in records if row["id"] in old_missing_ids and row["correct"]
            ),
            "remaining_previous_missing_failures": sorted(
                row["id"] for row in records if row["id"] in old_missing_ids and not row["correct"]
            ),
        },
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n逐条结果：{RESULTS_PATH}")
    print(f"汇总结果：{SUMMARY_PATH}")


if __name__ == "__main__":
    main()
