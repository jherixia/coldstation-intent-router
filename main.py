"""Interactive command-line entry point for the layered intent demo."""

from __future__ import annotations

import json

from router import LayeredIntentRouter


def main() -> None:
    print("正在加载本地 BGE 模型……")
    router = LayeredIntentRouter()
    print("分层意图识别 Demo 已启动。输入 exit 或 quit 退出。")

    while True:
        try:
            text = input("\n请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        if text.lower() in {"exit", "quit"}:
            print("已退出。")
            break
        if not text:
            print("输入不能为空。")
            continue

        result = router.route(text)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
