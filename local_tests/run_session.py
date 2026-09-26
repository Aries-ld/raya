"""右键 Run/Debug 一次；改 JSON/媒体后回车重测，同一模型可切换三种模态。"""

import sys
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parent))

from raya_maas.errors import APIError  # noqa: E402
from raya_maas.local_test import LocalDecisionSession, load_request_file  # noqa: E402


def main():
    selected = "image"
    print("正在加载模型，本次进程只加载一次……", flush=True)
    try:
        with LocalDecisionSession(DIRECTORY.parent) as session:
            print(
                f"模型已就绪：{session.engine.device}，加载 {session.model_load_ms:.0f} ms。\n"
                "首次运行 image.json；修改 JSON/媒体并保存后，回车重测。\n"
                "输入 text / image / video 切换并运行；输入 q 退出。",
                flush=True,
            )
            while True:
                path = DIRECTORY / f"{selected}.json"
                print(f"\n读取最新 {path.name}……", flush=True)
                try:
                    # Every iteration reloads JSON and media bytes, never a cached request/result.
                    request = load_request_file(path, session.settings)
                    result = session.predict(request)
                    # Inspect request/result here while debugging; Resume reaches the input prompt.
                    print(f"完成 {len(result['answers'])} 个决策，模型保持加载。", flush=True)
                except (OSError, ValueError, APIError) as exc:
                    print(f"输入错误，修改文件后可以重试：{exc}", flush=True)
                while True:
                    try:
                        command = input(
                            f"[{selected}] 回车重测 / text / image / video / q > "
                        ).strip()
                    except EOFError:
                        return
                    if command.lower() in ("q", "quit", "exit"):
                        return
                    if not command:
                        break
                    if command.lower() in ("text", "image", "video"):
                        selected = command.lower()
                        break
                    print("请输入 text、image、video、q，或直接回车。", flush=True)
    except KeyboardInterrupt:
        print("\n测试会话已结束。", flush=True)


if __name__ == "__main__":
    main()
