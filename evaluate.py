"""Evaluate the layered router on the fixed, manually labeled CSV set."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from router import LayeredIntentRouter


BASE_DIR = Path(__file__).resolve().parent
CASES_PATH = BASE_DIR / "eval_cases.csv"
RESULTS_PATH = BASE_DIR / "eval_results_qwen.csv"


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def main() -> None:
    with CASES_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        cases = list(csv.DictReader(source))

    router = LayeredIntentRouter()
    records = []
    for case in cases:
        result = router.route(case["text"])
        l2_trace = next(
            (step for step in result["trace"] if step["layer"] == "L2"),
            None,
        )
        records.append({
            "text": case["text"],
            "expected_intent": case["expected_intent"],
            "got_intent": result["final_intent"],
            "hit_layer": result["hit_layer"],
            "elapsed_seconds": result["total_seconds"],
            "correct": result["final_intent"] == case["expected_intent"],
            "called_small_llm": result["called_small_llm"],
            "l2_status": l2_trace.get("status", "") if l2_trace else "",
            "l2_intent": l2_trace.get("intent", "") if l2_trace else "",
            "l2_elapsed_seconds": l2_trace.get("elapsed_seconds", "") if l2_trace else "",
            "l2_raw_output": l2_trace.get("raw_output", "") if l2_trace else "",
            "fallback_reason": result["fallback_reason"] or "",
        })

    with RESULTS_PATH.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    total = len(records)
    layer_counts = Counter(record["hit_layer"] for record in records)
    l1_reached = [record for record in records if record["hit_layer"] != "L0"]
    l1_fallback_count = sum(record["hit_layer"] in {"L2", "L3"} for record in records)
    durations = [float(record["elapsed_seconds"]) for record in records]
    l2_status = Counter(
        record["fallback_reason"] for record in records if record["fallback_reason"]
    )

    summary = {
        "total_questions": total,
        "overall_accuracy": sum(record["correct"] for record in records) / total,
        "layer_hits": {
            layer: {
                "count": layer_counts[layer],
                "proportion": layer_counts[layer] / total,
            }
            for layer in ("L0", "L1", "L2", "L3")
        },
        "l1_fallback": {
            "count": l1_fallback_count,
            "proportion_of_all": l1_fallback_count / total,
            "proportion_of_l1_reached": l1_fallback_count / len(l1_reached) if l1_reached else "N/A",
        },
        "l2_call": {
            "count": sum(record["called_small_llm"] for record in records),
            "proportion": sum(record["called_small_llm"] for record in records) / total,
        },
        "l3_fallback": {
            "count": layer_counts["L3"],
            "proportion": layer_counts["L3"] / total,
        },
        "latency_seconds": {
            "average": statistics.fmean(durations),
            "p50": percentile(durations, 0.50),
            "p95": percentile(durations, 0.95),
        },
        "l2_fallback_status_counts": dict(l2_status),
        "complete_model_chain": not bool(l2_status.get("L2_UNAVAILABLE")),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n逐条结果：")
    for record in records:
        print(json.dumps(record, ensure_ascii=False))
    print(f"\nCSV 结果：{RESULTS_PATH}")


if __name__ == "__main__":
    main()
