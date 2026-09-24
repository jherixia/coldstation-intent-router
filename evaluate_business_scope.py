"""Evaluate the current layered demo against 20 cold-station business questions."""

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
CASES_PATH = BASE_DIR / "business_scope_eval_20.csv"
RESULTS_PATH = BASE_DIR / "business_scope_eval_20_results.csv"
SUMMARY_PATH = BASE_DIR / "business_scope_eval_20_summary.json"


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def main() -> None:
    with CASES_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        cases = list(csv.DictReader(source))
    if len(cases) != 20 or len({row["id"] for row in cases}) != 20:
        raise ValueError("business test set must contain 20 unique IDs")

    supported = [row for row in cases if row["support_status"] == "SUPPORTED"]
    unsupported = [row for row in cases if row["support_status"] == "NOT_IMPLEMENTED"]
    for row in supported:
        if row["expected_intent"] not in ALL_INTENTS:
            raise ValueError(f"unsupported expected intent in row {row['id']}")

    router = LayeredIntentRouter()
    records: list[dict[str, Any]] = []
    for case in cases:
        record: dict[str, Any] = dict(case)
        if case["support_status"] == "NOT_IMPLEMENTED":
            record.update({
                "actual_intent": "NOT_IMPLEMENTED",
                "hit_layer": "",
                "qwen_called": "",
                "total_seconds": "",
                "correct": "N/A",
                "simulated_response": "",
                "control_triggered": False,
                "note": "未送入模型，不计入已支持类别准确率",
            })
        else:
            result = router.route(case["text"])
            record.update({
                "actual_intent": result["final_intent"],
                "hit_layer": result["hit_layer"],
                "qwen_called": result["called_small_llm"],
                "total_seconds": result["total_seconds"],
                "correct": result["final_intent"] == case["expected_intent"],
                "simulated_response": result["simulated_response"],
                "control_triggered": False,
                "note": result["fallback_reason"] or "",
            })
        records.append(record)

    with RESULTS_PATH.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    executed = [row for row in records if row["support_status"] == "SUPPORTED"]
    errors = [row for row in executed if row["correct"] is False]
    durations = [float(row["total_seconds"]) for row in executed]
    qwen_calls = sum(bool(row["qwen_called"]) for row in executed)
    layer_counts = Counter(row["hit_layer"] for row in executed)

    opt_rows = [row for row in executed if row["expected_intent"] == "opt.read"]
    compare_rows = [row for row in executed if row["expected_intent"] == "mon.compare"]
    clarify_rows = [row for row in executed if row["expected_intent"] == "clarify"]
    control_rows = [row for row in executed if row["expected_intent"] == "reject_control"]
    mixed_control = next(row for row in executed if row["id"] == "16")

    summary = {
        "test_set": {
            "path": str(CASES_PATH),
            "sha256": hashlib.sha256(CASES_PATH.read_bytes()).hexdigest(),
            "total": len(records),
            "business_categories": list(dict.fromkeys(row["business_category"] for row in records)),
        },
        "support": {
            "supported_count": len(supported),
            "not_implemented_count": len(unsupported),
            "implemented_intents": list(ALL_INTENTS),
            "not_implemented": [
                {
                    "id": row["id"],
                    "text": row["text"],
                    "business_category": row["business_category"],
                    "expected_intent": row["expected_intent"],
                    "expected_handling": row["expected_handling"],
                }
                for row in unsupported
            ],
        },
        "executed_results": {
            "count": len(executed),
            "correct": sum(row["correct"] is True for row in executed),
            "accuracy": sum(row["correct"] is True for row in executed) / len(executed),
            "layer_hits": dict(layer_counts),
            "qwen_calls": qwen_calls,
            "latency_seconds": {
                "average": statistics.fmean(durations),
                "p50": percentile(durations, 0.50),
                "p95": percentile(durations, 0.95),
            },
            "errors": [
                {
                    "id": row["id"],
                    "text": row["text"],
                    "expected_intent": row["expected_intent"],
                    "actual_intent": row["actual_intent"],
                    "hit_layer": row["hit_layer"],
                }
                for row in errors
            ],
        },
        "checks": {
            "optimization_query_misread_as_navigation": [
                row["id"] for row in opt_rows if row["actual_intent"].startswith("nav.")
            ],
            "comparison_misread_as_plain_read": [
                row["id"] for row in compare_rows if row["actual_intent"] == "mon.read"
            ],
            "ambiguous_request_forced_into_business_intent": [
                row["id"] for row in clarify_rows if row["actual_intent"] != "clarify"
            ],
            "control_request_misread_as_query": [
                row["id"] for row in control_rows
                if row["actual_intent"] in {"mon.read", "mon.compare", "opt.read"}
            ],
            "mixed_query_control_rejected": mixed_control["actual_intent"] == "reject_control",
            "real_control_trigger_count": sum(bool(row["control_triggered"]) for row in records),
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
