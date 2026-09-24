"""Independent L0-L3 layered intent router demo.

This module reuses the local BGE model and the scoring approach from ../app.py.
It never calls a real business or device-control interface.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sentence_transformers import SentenceTransformer


OFFICIAL_BUSINESS_INTENTS = (
    "mon.read",
    "mon.compare",
    "opt.read",
    "nav.opt",
    "nav.overview",
    "nav.sim",
    "nav.history",
)
# Demo-internal handling names. The proof document specifies handling behavior,
# not official production intent codes, for these three paths.
INTERNAL_HANDLING_TYPES = (
    "internal.explain",
    "internal.savings_estimate",
    "internal.capability_missing",
)
BUSINESS_INTENTS = OFFICIAL_BUSINESS_INTENTS + INTERNAL_HANDLING_TYPES
ALL_INTENTS = BUSINESS_INTENTS + ("clarify", "reject_control")

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
MIN_SIMILARITY = 0.80
MIN_MARGIN = 0.06

INTENT_PROTOTYPES = {
    "mon.read": [
        "查看当前机组运行工况",
        "查询设备当前运行状态",
        "现在系统工况怎么样",
        "看一下当前生产运行情况",
    ],
    "mon.compare": [
        "对比两个时段的运行工况",
        "比较今天和昨天的设备状态",
        "查看不同工况之间的差异",
        "对比一号和二号机组运行情况",
    ],
    "opt.read": [
        "查看最新的寻优结果",
        "查询当前优化方案",
        "看看系统推荐的最优方案",
        "显示最近一次寻优方案",
    ],
    "nav.opt": [
        "打开寻优页面",
        "进入优化界面",
        "跳转到寻优页面",
        "带我去优化功能页面",
    ],
    "nav.overview": [
        "打开能效总览页面",
        "进入智能总览",
    ],
    "nav.sim": [
        "打开系统仿真中心",
        "进入仿真页面",
    ],
    "nav.history": [
        "打开寻优历史页面",
        "进入历史寻优记录页面",
    ],
    "internal.explain": [
        "解释冷站专业指标的含义",
        "湿球温度的定义是什么",
        "冷却塔逼近度是什么",
    ],
    "internal.savings_estimate": [
        "估算寻优方案可以节省的电费",
        "优化方案预计能省多少费用",
    ],
    "internal.capability_missing": [
        "源网荷储调度不在当前系统能力范围",
        "请求当前系统尚未接入的业务能力",
    ],
}


SIMULATED_RESPONSES = {
    "mon.read": "已识别为工况查询；实际系统将调用只读工况查询接口，本 Demo 不返回实时测点值。",
    "mon.compare": "已识别为工况对比；实际系统将调用只读历史/工况对比接口，本 Demo 不生成真实对比数据。",
    "opt.read": "已识别为寻优结果查询；实际系统将调用只读寻优方案接口，本 Demo 不生成真实寻优结果。",
    "nav.opt": "已识别为打开寻优页面；实际系统将执行前端页面导航，本 Demo 不操作真实业务系统。",
    "nav.overview": "已识别为打开智能总览；实际系统将执行前端页面导航，本 Demo 不操作真实业务系统。",
    "nav.sim": "已识别为打开仿真中心；实际系统将执行前端页面导航，本 Demo 不操作真实仿真系统。",
    "nav.history": "已识别为打开寻优历史页面；实际系统将执行前端页面导航，本 Demo 不查询真实历史数据。",
    "internal.explain": "已识别为机制或知识库说明请求；实际系统将查询说明模板或知识库，本 Demo 不接入真实知识库。",
    "internal.savings_estimate": "已识别为节能电费估算请求；实际系统需读取能耗、电价和寻优数据后估算。本 Demo 不生成估算数值，结果仅供参考。",
    "internal.capability_missing": "该业务能力当前未接入，接口后续接入中。",
    "clarify": "暂时无法明确你的意图，请说明是查询工况、对比工况、查看寻优结果，还是打开寻优页面。",
    "reject_control": "已识别为设备控制请求。本 Demo 不连接设备，也不会执行启动、停止或参数调整操作。",
}


DIRECT_CONTROL_ACTIONS = (
    "启动", "关闭", "关掉", "停止", "停掉", "停机", "切断", "复位", "下发", "写入",
    "开大", "关小", "调高", "调低", "调到", "改成", "设为", "提高到", "降低到",
)
AMBIGUOUS_CONTROL_ACTIONS = ("打开", "开启", "设置", "切换", "切到")
CONTROL_COMMAND_CUES = ("请", "帮我", "给我", "把", "将", "立即", "直接", "然后", "再把", "读完")
CONTROL_EXPLANATION_CUES = ("为什么", "为何", "什么是", "是什么意思", "原理", "机制")
CONTROL_COMPARISON_CUES = ("前后", "差异", "不同", "对比", "比较")
PHYSICAL_CONTROL_TARGETS = (
    "机组", "号机", "冷机", "水泵", "泵", "阀门", "阀", "电机", "风机", "开关", "断路器", "变频器",
)
ADJUST_ACTIONS = ("调高", "调低", "调到", "改成", "设为", "设置", "下发", "提高到", "降低到", "切到")
PARAMETER_TARGETS = ("频率", "转速", "功率", "温度", "压力", "设定值", "开度", "控制模式")
CONTROL_PLAN_ACTIONS = ("执行", "下发", "写入")
CONTROL_PLAN_TARGETS = ("开机组合", "启停组合", "控制方案", "推荐参数")
NAV_ACTIONS = ("打开", "进入", "跳转", "切换到", "带我去", "导航到")
NAV_TARGETS = ("寻优页面", "优化页面", "寻优界面", "优化界面", "寻优功能页面")
PAGE_NAV_ACTIONS = NAV_ACTIONS + ("去", "查看")
ADDITIONAL_NAVIGATION_TARGETS = {
    "nav.overview": ("智能总览", "能效总览", "总览页面"),
    "nav.sim": ("仿真中心", "仿真页面", "仿真模块"),
    "nav.history": ("寻优历史页面", "优化历史页面", "历史寻优页面"),
}
READ_ACTIONS = ("查看", "查询", "看看", "显示", "告诉我", "给我看", "调出")
OPT_RESULT_TARGETS = ("寻优结果", "优化结果", "寻优方案", "优化方案", "最优方案", "推荐方案")
OPT_DECISION_ACTIONS = ("采用", "选", "选择", "查看", "查询", "读取", "告诉我", "列出")
OPT_DECISION_TARGETS = ("寻优建议", "优化建议", "推荐模式", "寻优模式", "优化模式")
SAVINGS_ESTIMATE_ACTIONS = ("估算", "估计", "粗算", "预计", "大概", "能省", "节省", "少交")
SAVINGS_ESTIMATE_TARGETS = ("电费", "费用", "用电成本", "节电量", "多少钱")
CONDITION_QUERY_SCOPES = ("机房", "冷站")
CONDITION_QUERY_TARGETS = ("运行", "工况", "状态")
CONDITION_QUERY_CUES = ("怎么样", "如何", "情况", "状态", "查询", "查看", "看看")


def _contains_pair(text: str, actions: tuple[str, ...], targets: tuple[str, ...]) -> bool:
    return any(action in text for action in actions) and any(target in text for target in targets)


def classify_l0(text: str) -> tuple[str | None, str]:
    """Return an intent only for explicit, high-precision rules."""
    explanatory_context = any(cue in text for cue in CONTROL_EXPLANATION_CUES)
    comparison_context = "前后" in text and any(cue in text for cue in CONTROL_COMPARISON_CUES)
    direct_command = (
        any(cue in text for cue in CONTROL_COMMAND_CUES)
        or any(text.startswith(action) for action in DIRECT_CONTROL_ACTIONS + AMBIGUOUS_CONTROL_ACTIONS)
    )
    explicit_physical_control = direct_command and _contains_pair(
        text,
        DIRECT_CONTROL_ACTIONS + AMBIGUOUS_CONTROL_ACTIONS,
        PHYSICAL_CONTROL_TARGETS,
    )
    explicit_generic_control = direct_command and _contains_pair(
        text,
        DIRECT_CONTROL_ACTIONS + AMBIGUOUS_CONTROL_ACTIONS,
        ("设备",),
    )
    explicit_parameter_control = direct_command and _contains_pair(text, ADJUST_ACTIONS, PARAMETER_TARGETS)
    explicit_control_plan = _contains_pair(text, CONTROL_PLAN_ACTIONS, CONTROL_PLAN_TARGETS)
    if (
        not explanatory_context
        and not comparison_context
        and (explicit_physical_control or explicit_generic_control or explicit_parameter_control or explicit_control_plan)
    ):
        return "reject_control", "explicit_control_action_and_target"
    for intent, targets in ADDITIONAL_NAVIGATION_TARGETS.items():
        if _contains_pair(text, PAGE_NAV_ACTIONS, targets):
            return intent, "explicit_navigation_action_and_page"
    if _contains_pair(text, NAV_ACTIONS, NAV_TARGETS):
        return "nav.opt", "explicit_navigation_action_and_page"
    if _contains_pair(text, SAVINGS_ESTIMATE_ACTIONS, SAVINGS_ESTIMATE_TARGETS):
        return "internal.savings_estimate", "explicit_savings_estimate_request"
    if _contains_pair(text, READ_ACTIONS, OPT_RESULT_TARGETS):
        return "opt.read", "explicit_read_action_and_optimization_result"
    if _contains_pair(text, OPT_DECISION_ACTIONS, OPT_DECISION_TARGETS):
        return "opt.read", "explicit_read_only_optimization_decision"
    explicit_condition_query = (
        any(scope in text for scope in CONDITION_QUERY_SCOPES)
        and any(target in text for target in CONDITION_QUERY_TARGETS)
        and any(cue in text for cue in CONDITION_QUERY_CUES)
    )
    if explicit_condition_query:
        return "mon.read", "explicit_cold_station_condition_query"
    return None, "no_explicit_rule_match"


@dataclass
class L2Config:
    api_url: str | None
    model: str | None
    api_key: str | None
    timeout_seconds: float = 30.0

    @classmethod
    def from_environment(cls) -> "L2Config":
        return cls(
            api_url=os.getenv(
                "L2_API_URL",
                "http://127.0.0.1:11434/v1/chat/completions",
            ),
            model=os.getenv("L2_MODEL", "qwen2.5:1.5b"),
            api_key=os.getenv("L2_API_KEY"),
            timeout_seconds=float(os.getenv("L2_TIMEOUT_SECONDS", "30")),
        )

    @property
    def available(self) -> bool:
        return bool(self.api_url and self.model)


class LayeredIntentRouter:
    def __init__(self, l2_config: L2Config | None = None) -> None:
        model_path = os.getenv("BGE_MODEL_PATH", MODEL_NAME)
        self.model = SentenceTransformer(
            model_path,
            device="cpu",
            local_files_only=True,
        )
        self.l2_config = l2_config or L2Config.from_environment()

        self.prototype_texts: list[str] = []
        self.prototype_intents: list[str] = []
        for intent, prototypes in INTENT_PROTOTYPES.items():
            for prototype in prototypes:
                self.prototype_texts.append(prototype)
                self.prototype_intents.append(intent)

        self.prototype_vectors = self.model.encode(
            self.prototype_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def classify_l1(self, text: str) -> dict[str, Any]:
        query_vector = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        similarities = self.prototype_vectors @ query_vector
        scores = {
            intent: max(
                float(similarities[index])
                for index, prototype_intent in enumerate(self.prototype_intents)
                if prototype_intent == intent
            )
            for intent in BUSINESS_INTENTS
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        candidate, similarity = ranked[0]
        margin = similarity - ranked[1][1]
        accepted = similarity >= MIN_SIMILARITY and margin >= MIN_MARGIN
        return {
            "candidate": candidate,
            "similarity": round(similarity, 4),
            "margin": round(margin, 4),
            "accepted": accepted,
        }

    def classify_l2(self, text: str) -> dict[str, Any]:
        config = self.l2_config
        if not config.available:
            return {"status": "L2_UNAVAILABLE", "intent": None, "called": False}

        system_prompt = (
            "你是冷站能效提升系统的意图分类器。只根据用户原话分类，不回答问题，不补充上下文，不编造业务数据。\n"
            "分类边界：\n"
            "1. 默认选择 clarify。只有原话明确包含对应业务对象和请求动作时，才能选择其他代码。\n"
            "2. 缺少业务对象、只有代词或泛泛询问如何处理/下一步怎么办时，必须选择 clarify。\n"
            "3. 设备控制优先级最高。只要一句话中包含启动、停止、开关设备或修改参数，即使同时要求查询，也选择 reject_control；一号机、1号机等编号设备属于具体设备。\n"
            "4. COP、湿球温度、温度、压力等冷站指标的当前值或历史值查询属于 mon.read。\n"
            "5. 比较两个指标、两个时段、两台设备或两种工况，以及询问它们的区别，属于 mon.compare。\n"
            "6. 获取本轮推荐方案、推荐模式、优化建议或寻优结果属于 opt.read；即使在模式A和模式B之间询问建议，也不属于导航。\n"
            "7. 只有明确要求打开、进入、跳转或切换到具体页面时，才能选择对应 nav.*；查询或展示方案内容不能选择 nav.*。\n"
            "8. 询问机房当前运行情况属于 mon.read；只问系统能效怎么样但没有明确指标、时段或范围时，选择 clarify。两者不要混淆。\n"
            "9. 询问专业概念、原理或术语含义时选择 internal.explain。\n"
            "10. 请求估算寻优方案节省的电量或电费时选择 internal.savings_estimate。\n"
            "11. 源网荷储等当前系统未接入的业务能力选择 internal.capability_missing。\n"
            "只能选择以下代码：\n"
            "mon.read：查询冷站指标读数、当前或历史工况；\n"
            "mon.compare：比较指标、时段、设备或工况；\n"
            "opt.read：只读查看寻优结果、推荐模式、推荐方案或优化建议；\n"
            "nav.opt：请求打开、进入或跳转到寻优页面；\n"
            "nav.overview：请求打开、进入或跳转到智能总览页面；\n"
            "nav.sim：请求打开、进入或跳转到仿真中心；\n"
            "nav.history：请求打开、进入或跳转到寻优历史页面；\n"
            "internal.explain：Demo内部处理类型，表示机制、术语或知识库说明；\n"
            "internal.savings_estimate：Demo内部处理类型，表示节能电费估算；\n"
            "internal.capability_missing：Demo内部处理类型，表示当前能力未接入；\n"
            "reject_control：请求启动、停止、开关设备或修改设备参数；\n"
            "clarify：信息不足或缺少明确查询目标，需要用户澄清。\n"
            "分类示例：\n"
            "室外湿球是多少 -> mon.read\n"
            "两个能效指标有什么区别 -> mon.compare\n"
            "推荐模式选哪个 -> opt.read\n"
            "打开寻优页面 -> nav.opt\n"
            "机房目前运行情况如何 -> mon.read\n"
            "系统能效怎么样（未说明指标、时段或范围） -> clarify\n"
            "把一号机停了 -> reject_control\n"
            "不得输出中文意图描述，只能输出上述代码之一。"
        )
        payload = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": 0,
            "seed": 0,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "intent_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "intent": {
                                "type": "string",
                                "enum": list(ALL_INTENTS),
                            }
                        },
                        "required": ["intent"],
                        "additionalProperties": False,
                    },
                },
            },
        }
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        request = Request(
            config.api_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urlopen(request, timeout=config.timeout_seconds) as response:
                response_body = json.loads(response.read().decode("utf-8"))
            content = response_body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            if set(parsed) != {"intent"} or parsed["intent"] not in ALL_INTENTS:
                raise ValueError("invalid intent JSON")
            intent = parsed["intent"]
            status = "L2_UNCERTAIN" if intent == "clarify" else "L2_OK"
            return {
                "status": status,
                "intent": intent,
                "called": True,
                "raw_output": content,
            }
        except (HTTPError, URLError, TimeoutError, KeyError, IndexError, TypeError,
                ValueError, json.JSONDecodeError) as exc:
            return {
                "status": "L2_INVALID_OUTPUT" if isinstance(exc, (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError)) else "L2_UNAVAILABLE",
                "intent": None,
                "called": True,
                "error": str(exc),
            }

    def route(self, text: str) -> dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("input text must not be empty")

        started = time.perf_counter()
        trace: list[dict[str, Any]] = []

        l0_started = time.perf_counter()
        l0_intent, l0_reason = classify_l0(text)
        trace.append({
            "layer": "L0",
            "matched": l0_intent is not None,
            "intent": l0_intent,
            "reason": l0_reason,
            "elapsed_seconds": round(time.perf_counter() - l0_started, 6),
        })
        if l0_intent:
            return self._result(text, l0_intent, "L0", trace, started, False, None)

        l1_started = time.perf_counter()
        l1 = self.classify_l1(text)
        l1["layer"] = "L1"
        l1["elapsed_seconds"] = round(time.perf_counter() - l1_started, 6)
        trace.append(l1)
        if l1["accepted"]:
            return self._result(text, l1["candidate"], "L1", trace, started, False, None)

        l2_started = time.perf_counter()
        l2 = self.classify_l2(text)
        l2["layer"] = "L2"
        l2["elapsed_seconds"] = round(time.perf_counter() - l2_started, 6)
        trace.append(l2)
        if l2["status"] == "L2_OK":
            return self._result(text, l2["intent"], "L2", trace, started, l2["called"], None)

        fallback_reason = l2["status"]
        trace.append({
            "layer": "L3",
            "triggered": True,
            "reason": fallback_reason,
            "intent": "clarify",
            "elapsed_seconds": 0.0,
        })
        return self._result(
            text, "clarify", "L3", trace, started, l2["called"], fallback_reason
        )

    @staticmethod
    def _result(
        text: str,
        intent: str,
        hit_layer: str,
        trace: list[dict[str, Any]],
        started: float,
        called_small_llm: bool,
        fallback_reason: str | None,
    ) -> dict[str, Any]:
        return {
            "text": text,
            "final_intent": intent,
            "hit_layer": hit_layer,
            "trace": trace,
            "total_seconds": round(time.perf_counter() - started, 6),
            "called_small_llm": called_small_llm,
            "fallback_reason": fallback_reason,
            "simulated_response": SIMULATED_RESPONSES[intent],
        }
