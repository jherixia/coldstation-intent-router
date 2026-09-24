"""Run the fixed 100-question cold-station holdout exactly once.

The expected intent is used only after routing for evaluation. It is never passed
to LayeredIntentRouter or any classification layer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from router import ALL_INTENTS, LayeredIntentRouter, MIN_MARGIN, MIN_SIMILARITY


BASE_DIR = Path(__file__).resolve().parent
CASES_PATH = BASE_DIR / "coldstation_intent_100_holdout.csv"
RESULTS_PATH = BASE_DIR / "coldstation_intent_100_results.csv"
SUMMARY_PATH = BASE_DIR / "coldstation_intent_100_summary.json"
REPORT_PATH = BASE_DIR / "coldstation_intent_100_report.md"
EXPECTED_SHA256 = "4a10f9e8c170fd842e65b8801a6bdb98a9b98fecef341b18878c801a7f4c95b6"
L2_FAILURE_STATUSES = {"L2_UNAVAILABLE", "L2_INVALID_OUTPUT"}


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def percentage(count: int, total: int) -> float | None:
    return count / total if total else None


def handling_check(expected: str, actual: str, response: str) -> tuple[bool, str]:
    if actual != expected:
        return False, "最终意图与人工标注不一致"
    if expected == "internal.explain":
        ok = "机制" in response and "知识库" in response
        return ok, "" if ok else "机制说明模拟响应未明确知识库/说明性处理边界"
    if expected == "internal.savings_estimate":
        ok = "仅供参考" in response and "不生成估算数值" in response
        return ok, "" if ok else "电费估算响应缺少不生成数值或仅供参考说明"
    if expected == "internal.capability_missing":
        ok = "接口后续接入中" in response
        return ok, "" if ok else "域外能力响应未说明接口后续接入"
    if expected == "reject_control":
        ok = "不会执行" in response
        return ok, "" if ok else "拒控响应未明确不会执行设备控制"
    return True, ""


def trace_step(result: dict[str, Any], layer: str) -> dict[str, Any] | None:
    return next((step for step in result["trace"] if step["layer"] == layer), None)


def validate_cases() -> tuple[list[dict[str, str]], str]:
    if not CASES_PATH.exists():
        raise FileNotFoundError(CASES_PATH)
    digest = hashlib.sha256(CASES_PATH.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"holdout SHA-256 mismatch: expected {EXPECTED_SHA256}, got {digest}")
    with CASES_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["text", "expected_intent", "label_basis"]:
            raise ValueError(f"unexpected CSV columns: {reader.fieldnames}")
        cases = list(reader)
    if len(cases) != 100:
        raise ValueError(f"holdout must contain 100 rows, got {len(cases)}")
    if any(not value.strip() for row in cases for value in row.values()):
        raise ValueError("holdout contains a blank required field")
    if len({row["text"] for row in cases}) != len(cases):
        raise ValueError("holdout contains duplicate question text")
    labels = {row["expected_intent"] for row in cases}
    supported = set(ALL_INTENTS)
    if labels != supported:
        raise ValueError(
            f"holdout labels must cover exactly the supported intents; "
            f"missing={sorted(supported - labels)}, extra={sorted(labels - supported)}"
        )
    return cases, digest


def build_report(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    lines = [
        "# 冷站能效分层意图识别 Demo：100 条新问法首次评测",
        "",
        "## 测试条件",
        "",
        f"- 测试集：`coldstation_intent_100_holdout.csv`，SHA-256 `{summary['test_set']['sha256']}`。",
        "- 共 100 条，覆盖当前 Demo 全部 12 种意图及处理类型。",
        f"- 阈值保持 `MIN_SIMILARITY={summary['configuration']['min_similarity']:.2f}`、"
        f"`MIN_MARGIN={summary['configuration']['min_margin']:.2f}`。",
        "- 未进行模型预热。BGE 路由器初始化耗时单独记录并排除在 100 条请求耗时之外；首次 Qwen 调用耗时保留在正式结果中。",
        "- `expected_intent` 仅用于路由完成后的统计，没有传入路由器、提示词或任何分类层。",
        "",
        "## 总体结果",
        "",
        f"- 正确 {overall['correct']}/{overall['total']}，准确率 {overall['accuracy']:.2%}。",
        f"- 平均端到端耗时 {overall['latency_seconds']['average']:.6f} 秒，"
        f"p50 {overall['latency_seconds']['p50']:.6f} 秒，"
        f"p95 {overall['latency_seconds']['p95']:.6f} 秒。",
        f"- Qwen 调用 {summary['qwen']['calls']} 次（{summary['qwen']['call_ratio']:.2%}），"
        f"平均调用耗时 {summary['qwen']['average_latency_seconds']:.6f} 秒。",
        f"- L2 不可用/无效输出 {summary['qwen']['failure_count']} 次。",
        f"- 实际设备控制调用 {summary['safety']['actual_control_trigger_count']} 次。",
        "",
        "## 各意图结果",
        "",
        "| 期望意图 | 总数 | 正确 | 错误 | 准确率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for intent in ALL_INTENTS:
        item = summary["per_intent"][intent]
        lines.append(
            f"| {intent} | {item['total']} | {item['correct']} | {item['errors']} | {item['accuracy']:.2%} |"
        )
    lines.extend([
        "",
        "## 各层最终命中",
        "",
        "| 层级 | 数量 | 比例 |",
        "|---|---:|---:|",
    ])
    for layer in ("L0", "L1", "L2", "L3"):
        item = summary["layer_hits"][layer]
        lines.append(f"| {layer} | {item['count']} | {item['ratio']:.2%} |")

    focus = summary["focus_checks"]
    lines.extend([
        "",
        "## 重点检查",
        "",
        f"- `mon.read` 与 `mon.compare` 相互混淆：{focus['read_compare_confusions']} 条。",
        f"- L1 错误提前结束、导致 L2 无法纠正：{focus['l1_early_errors']} 条。",
        f"- `opt.read` 与 `nav.opt` 相互混淆：{focus['opt_navigation_confusions']} 条。",
        f"- `clarify` 被强行分类成具体业务意图：{focus['clarify_forced_specific']} 条。",
        f"- 12 条拒控请求正确识别 {summary['safety']['reject_correct']} 条，漏识别 {summary['safety']['reject_missed']} 条。",
        f"- 查询与控制混合请求漏控：{summary['safety']['mixed_control_missed']} 条。",
        f"- 新增内部处理类型模拟响应边界违规：{focus['internal_response_violations']} 条。",
        "",
        "## 冷启动记录",
        "",
        f"- 路由器/BGE 初始化：{summary['cold_start']['router_initialization_seconds']:.6f} 秒（不计入正式请求耗时）。",
        f"- 首条正式请求：{summary['cold_start']['first_request_seconds']:.6f} 秒。",
    ])
    first_qwen = summary["cold_start"]["first_qwen_call"]
    if first_qwen:
        lines.append(
            f"- 首次 Qwen 调用为第 {first_qwen['row_number']} 条，调用耗时 "
            f"{first_qwen['elapsed_seconds']:.6f} 秒；该耗时保留在正式统计中。"
        )

    lines.extend(["", "## 全部错误案例", ""])
    if not summary["errors"]:
        lines.append("无。")
    else:
        lines.extend([
            "| 序号 | 用户问题 | 期望 | 实际 | 错误层级 | 原因 |",
            "|---:|---|---|---|---|---|",
        ])
        for error in summary["errors"]:
            text = error["text"].replace("|", "\\|")
            reason = error["error_reason"].replace("|", "\\|")
            lines.append(
                f"| {error['row_number']} | {text} | {error['expected_intent']} | "
                f"{error['actual_intent']} | {error['error_layer']} | {reason} |"
            )

    lines.extend([
        "",
        "## 结论",
        "",
        summary["conclusion"],
        "",
        "本文件保存的是该新测试集的首次评测结果。评测后未修改提示词、规则、原型、阈值或模型。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    cases, digest = validate_cases()
    existing = [path for path in (RESULTS_PATH, SUMMARY_PATH, REPORT_PATH) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite first-evaluation outputs: {existing}")

    init_started = time.perf_counter()
    router = LayeredIntentRouter()
    router_initialization_seconds = time.perf_counter() - init_started

    records: list[dict[str, Any]] = []
    for row_number, case in enumerate(cases, start=1):
        # Ground truth is deliberately not passed to the router.
        result = router.route(case["text"])
        l0 = trace_step(result, "L0")
        l1 = trace_step(result, "L1")
        l2 = trace_step(result, "L2")
        l3 = trace_step(result, "L3")
        l2_status = l2.get("status", "") if l2 else ""
        model_failure = l2_status in L2_FAILURE_STATUSES
        handling_ok, handling_reason = handling_check(
            case["expected_intent"], result["final_intent"], result["simulated_response"]
        )
        correct = handling_ok and not model_failure
        if model_failure:
            error_reason = f"{l2_status}，实际回退为 {result['final_intent']}，不按正常澄清计为成功"
            error_layer = "L2"
        elif not handling_ok:
            error_reason = handling_reason
            error_layer = result["hit_layer"]
        else:
            error_reason = ""
            error_layer = ""
        records.append({
            "row_number": row_number,
            "text": case["text"],
            "expected_intent": case["expected_intent"],
            "label_basis": case["label_basis"],
            "actual_intent": result["final_intent"],
            "hit_layer": result["hit_layer"],
            "l0_matched": l0.get("matched", "") if l0 else "",
            "l0_intent": l0.get("intent", "") if l0 else "",
            "l0_reason": l0.get("reason", "") if l0 else "",
            "l1_candidate": l1.get("candidate", "") if l1 else "",
            "l1_similarity": l1.get("similarity", "") if l1 else "",
            "l1_margin": l1.get("margin", "") if l1 else "",
            "l1_accepted": l1.get("accepted", "") if l1 else "",
            "qwen_called": bool(l2 and l2.get("called")),
            "l2_status": l2_status,
            "qwen_output": l2.get("raw_output", "") if l2 else "",
            "qwen_intent": l2.get("intent", "") if l2 else "",
            "qwen_error": l2.get("error", "") if l2 else "",
            "qwen_elapsed_seconds": l2.get("elapsed_seconds", "") if l2 else "",
            "l3_triggered": bool(l3),
            "l3_fallback_reason": l3.get("reason", "") if l3 else "",
            "total_seconds": result["total_seconds"],
            "simulated_response": result["simulated_response"],
            "correct": correct,
            "error_layer": error_layer,
            "error_reason": error_reason,
            "actual_device_control_triggered": False,
        })

    with RESULTS_PATH.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    total = len(records)
    durations = [float(row["total_seconds"]) for row in records]
    layer_counts = Counter(str(row["hit_layer"]) for row in records)
    qwen_rows = [row for row in records if row["qwen_called"]]
    qwen_durations = [float(row["qwen_elapsed_seconds"]) for row in qwen_rows]
    errors = [row for row in records if not row["correct"]]

    per_intent: dict[str, dict[str, Any]] = {}
    for intent in ALL_INTENTS:
        subset = [row for row in records if row["expected_intent"] == intent]
        correct_count = sum(bool(row["correct"]) for row in subset)
        per_intent[intent] = {
            "total": len(subset),
            "correct": correct_count,
            "errors": len(subset) - correct_count,
            "accuracy": percentage(correct_count, len(subset)),
        }

    reject_rows = [row for row in records if row["expected_intent"] == "reject_control"]
    mixed_control_rows = [row for row in reject_rows if "查询与控制混合" in row["label_basis"] and any(
        cue in row["text"] for cue in ("先", "然后", "再", "读完")
    )]
    first_qwen = qwen_rows[0] if qwen_rows else None
    summary: dict[str, Any] = {
        "test_set": {
            "path": str(CASES_PATH),
            "sha256": digest,
            "total": total,
            "labels": dict(Counter(row["expected_intent"] for row in records)),
        },
        "configuration": {
            "all_intents": list(ALL_INTENTS),
            "min_similarity": MIN_SIMILARITY,
            "min_margin": MIN_MARGIN,
            "warmup_performed": False,
        },
        "overall": {
            "total": total,
            "correct": sum(bool(row["correct"]) for row in records),
            "errors": len(errors),
            "accuracy": percentage(sum(bool(row["correct"]) for row in records), total),
            "latency_seconds": {
                "average": statistics.fmean(durations),
                "p50": percentile(durations, 0.50),
                "p95": percentile(durations, 0.95),
            },
        },
        "per_intent": per_intent,
        "layer_hits": {
            layer: {"count": layer_counts.get(layer, 0), "ratio": percentage(layer_counts.get(layer, 0), total)}
            for layer in ("L0", "L1", "L2", "L3")
        },
        "qwen": {
            "calls": len(qwen_rows),
            "call_ratio": percentage(len(qwen_rows), total),
            "average_latency_seconds": statistics.fmean(qwen_durations) if qwen_durations else None,
            "failure_count": sum(row["l2_status"] in L2_FAILURE_STATUSES for row in qwen_rows),
            "status_counts": dict(Counter(str(row["l2_status"]) for row in qwen_rows)),
        },
        "cold_start": {
            "warmup_performed": False,
            "router_initialization_seconds": router_initialization_seconds,
            "first_request_seconds": float(records[0]["total_seconds"]),
            "first_qwen_call": ({
                "row_number": first_qwen["row_number"],
                "elapsed_seconds": float(first_qwen["qwen_elapsed_seconds"]),
            } if first_qwen else None),
        },
        "focus_checks": {
            "read_compare_confusions": sum(
                (row["expected_intent"], row["actual_intent"])
                in {("mon.read", "mon.compare"), ("mon.compare", "mon.read")}
                for row in records
            ),
            "l1_early_errors": sum(not row["correct"] and row["hit_layer"] == "L1" for row in records),
            "opt_navigation_confusions": sum(
                (row["expected_intent"], row["actual_intent"])
                in {("opt.read", "nav.opt"), ("nav.opt", "opt.read")}
                for row in records
            ),
            "clarify_forced_specific": sum(
                row["expected_intent"] == "clarify" and row["actual_intent"] != "clarify"
                for row in records
            ),
            "internal_response_violations": sum(
                row["expected_intent"].startswith("internal.")
                and row["actual_intent"] == row["expected_intent"]
                and not row["correct"]
                for row in records
            ),
        },
        "safety": {
            "reject_total": len(reject_rows),
            "reject_correct": sum(row["actual_intent"] == "reject_control" and row["correct"] for row in reject_rows),
            "reject_missed": sum(row["actual_intent"] != "reject_control" for row in reject_rows),
            "mixed_control_total": len(mixed_control_rows),
            "mixed_control_missed": sum(row["actual_intent"] != "reject_control" for row in mixed_control_rows),
            "actual_control_trigger_count": sum(bool(row["actual_device_control_triggered"]) for row in records),
        },
        "errors": [
            {
                "row_number": row["row_number"],
                "text": row["text"],
                "expected_intent": row["expected_intent"],
                "actual_intent": row["actual_intent"],
                "hit_layer": row["hit_layer"],
                "error_layer": row["error_layer"],
                "error_reason": row["error_reason"],
                "l1_candidate": row["l1_candidate"],
                "l1_similarity": row["l1_similarity"],
                "l1_margin": row["l1_margin"],
                "l2_status": row["l2_status"],
                "qwen_output": row["qwen_output"],
            }
            for row in errors
        ],
    }
    summary["conclusion"] = (
        "本次独立测试未发现错误；当前 Demo 在这 100 条固定新问法上满足预期。"
        if not errors
        else f"本次独立测试发现 {len(errors)} 条错误，需要结合错误层级审查，但本轮未修改算法。"
    )

    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n逐条结果：{RESULTS_PATH}")
    print(f"汇总结果：{SUMMARY_PATH}")
    print(f"中文报告：{REPORT_PATH}")


if __name__ == "__main__":
    main()
