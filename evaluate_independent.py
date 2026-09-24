"""Run independent A/B/C evaluations on the fixed 48-question test set."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from router import ALL_INTENTS, LayeredIntentRouter


EVAL_INTENTS = (
    "mon.read",
    "mon.compare",
    "opt.read",
    "nav.opt",
    "clarify",
    "reject_control",
)


BASE_DIR = Path(__file__).resolve().parent
CASES_PATH = BASE_DIR / "independent_eval_48.csv"
OLD_CASES_PATH = BASE_DIR / "eval_cases.csv"
RESULTS_PATH = BASE_DIR / "independent_eval_48_results.csv"
SUMMARY_PATH = BASE_DIR / "independent_eval_48_summary.json"


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def validate_cases(cases: list[dict[str, str]]) -> None:
    if len(cases) != 48:
        raise ValueError(f"expected 48 cases, got {len(cases)}")
    texts = [case["text"].strip() for case in cases]
    if any(not text for text in texts) or len(set(texts)) != len(texts):
        raise ValueError("test questions must be non-empty and unique")
    counts = Counter(case["expected_intent"] for case in cases)
    expected_counts = {intent: 8 for intent in EVAL_INTENTS}
    if counts != expected_counts:
        raise ValueError(f"expected 8 cases per intent, got {dict(counts)}")
    if any(not case["label_basis"].strip() for case in cases):
        raise ValueError("every case must include a label_basis")

    with OLD_CASES_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        old_texts = {row["text"].strip() for row in csv.DictReader(source)}
    duplicates = sorted(set(texts) & old_texts)
    if duplicates:
        raise ValueError(f"new set duplicates old questions: {duplicates}")


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "average": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }


def summarize_experiment(
    records: list[dict[str, Any]],
    output_field: str,
    correct_field: str,
    latency_field: str,
    layer_field: str | None = None,
) -> dict[str, Any]:
    class_metrics = {}
    for intent in EVAL_INTENTS:
        class_rows = [row for row in records if row["expected_intent"] == intent]
        correct = sum(bool(row[correct_field]) for row in class_rows)
        class_metrics[intent] = {
            "correct": correct,
            "errors": len(class_rows) - correct,
            "total": len(class_rows),
        }

    errors = []
    for row in records:
        if not row[correct_field]:
            error = {
                "text": row["text"],
                "expected_intent": row["expected_intent"],
                "got_intent": row[output_field],
            }
            if layer_field:
                error["hit_layer"] = row[layer_field]
            errors.append(error)

    return {
        "accuracy": sum(bool(row[correct_field]) for row in records) / len(records),
        "per_class": class_metrics,
        "clarify": class_metrics["clarify"],
        "reject_control": class_metrics["reject_control"],
        "latency_seconds": latency_summary(
            [float(row[latency_field]) for row in records]
        ),
        "errors": errors,
    }


def main() -> None:
    with CASES_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        cases = list(csv.DictReader(source))
    validate_cases(cases)
    test_set_hash = hashlib.sha256(CASES_PATH.read_bytes()).hexdigest()

    router = LayeredIntentRouter()
    records: list[dict[str, Any]] = [
        {
            "text": case["text"],
            "expected_intent": case["expected_intent"],
            "label_basis": case["label_basis"],
        }
        for case in cases
    ]

    # Experiment A: L1 BGE only. Rejected candidates become clarify.
    for row in records:
        started = time.perf_counter()
        l1 = router.classify_l1(row["text"])
        elapsed = time.perf_counter() - started
        output = l1["candidate"] if l1["accepted"] else "clarify"
        row.update({
            "a_intent": output,
            "a_correct": output == row["expected_intent"],
            "a_candidate": l1["candidate"],
            "a_similarity": l1["similarity"],
            "a_margin": l1["margin"],
            "a_accepted": l1["accepted"],
            "a_elapsed_seconds": round(elapsed, 6),
        })

    # Experiment B: L2 Qwen only. Invalid/unavailable output becomes clarify.
    for row in records:
        started = time.perf_counter()
        l2 = router.classify_l2(row["text"])
        elapsed = time.perf_counter() - started
        output = l2.get("intent") if l2.get("intent") in ALL_INTENTS else "clarify"
        row.update({
            "b_intent": output,
            "b_correct": output == row["expected_intent"],
            "b_status": l2["status"],
            "b_raw_output": l2.get("raw_output", ""),
            "b_elapsed_seconds": round(elapsed, 6),
        })

    # Experiment C: unchanged complete L0 -> L1 -> L2 -> L3 router.
    for row in records:
        result = router.route(row["text"])
        l2_trace = next(
            (step for step in result["trace"] if step["layer"] == "L2"),
            None,
        )
        correct = result["final_intent"] == row["expected_intent"]
        row.update({
            "c_intent": result["final_intent"],
            "c_correct": correct,
            "c_hit_layer": result["hit_layer"],
            "c_elapsed_seconds": result["total_seconds"],
            "c_l2_called": result["called_small_llm"],
            "c_l2_status": l2_trace.get("status", "") if l2_trace else "",
            "c_l2_intent": l2_trace.get("intent", "") if l2_trace else "",
            "c_l2_elapsed_seconds": l2_trace.get("elapsed_seconds", "") if l2_trace else "",
            "c_fallback_reason": result["fallback_reason"] or "",
            "c_error_origin_layer": "" if correct else result["hit_layer"],
            "c_early_layer_blocked_correction": (
                not correct and result["hit_layer"] in {"L0", "L1"}
            ),
        })

    with RESULTS_PATH.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "test_set": {
            "path": str(CASES_PATH),
            "sha256": test_set_hash,
            "total": len(records),
            "per_class": dict(Counter(row["expected_intent"] for row in records)),
        },
        "experiment_a_bge_only": summarize_experiment(
            records, "a_intent", "a_correct", "a_elapsed_seconds"
        ),
        "experiment_b_qwen_only": summarize_experiment(
            records, "b_intent", "b_correct", "b_elapsed_seconds"
        ),
        "experiment_c_layered": summarize_experiment(
            records,
            "c_intent",
            "c_correct",
            "c_elapsed_seconds",
            "c_hit_layer",
        ),
    }

    layer_counts = Counter(row["c_hit_layer"] for row in records)
    l2_rows = [row for row in records if row["c_l2_called"]]
    c_summary = summary["experiment_c_layered"]
    c_summary.update({
        "layer_hits": {
            layer: {
                "count": layer_counts[layer],
                "proportion": layer_counts[layer] / len(records),
            }
            for layer in ("L0", "L1", "L2", "L3")
        },
        "l2_calls": {
            "count": len(l2_rows),
            "proportion": len(l2_rows) / len(records),
            "average_inference_seconds": (
                statistics.fmean(float(row["c_l2_elapsed_seconds"]) for row in l2_rows)
                if l2_rows else "N/A"
            ),
        },
        "error_origin_layers": dict(
            Counter(
                row["c_error_origin_layer"]
                for row in records
                if row["c_error_origin_layer"]
            )
        ),
        "early_layer_blocked_correction_count": sum(
            bool(row["c_early_layer_blocked_correction"]) for row in records
        ),
    })

    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n逐条结果：{RESULTS_PATH}")
    print(f"汇总结果：{SUMMARY_PATH}")


if __name__ == "__main__":
    main()
