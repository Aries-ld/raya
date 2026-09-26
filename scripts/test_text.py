"""直接运行，或修改下方 QUESTION / CANDIDATES；也支持命令行参数。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from raya_maas.local_test import run_local_test  # noqa: E402

QUESTION = "用户说：帮我把明天下午3点的会议改到4点。这个请求的意图是什么？"
CANDIDATES = ["修改会议时间", "创建新会议", "取消会议", "查询会议详情"]
DEVICE = None  # None 读取 .env/默认配置，也可改成 "auto"、"cuda"、"mps"、"cpu"。
THRESHOLD = 0.6


def main():
    # 在这里打断点，Step Into 可进入模型加载与推理；result 是完整决策结果。
    result = run_local_test(
        "text",
        QUESTION,
        CANDIDATES,
        project_root=PROJECT_ROOT,
        device=DEVICE,
        threshold=THRESHOLD,
    )
    return result


if __name__ == "__main__":
    main()
