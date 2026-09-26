"""直接运行，或修改下方 VIDEO_PATH / QUESTION / CANDIDATES。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from raya_maas.local_test import run_local_test  # noqa: E402

VIDEO_PATH = PROJECT_ROOT / "tests/fixtures/m3bench_kitchen.mp4"
QUESTION = "Where does this video take place?"
CANDIDATES = ["A kitchen", "A beach", "An office", "A street"]
DEVICE = None  # None 读取 .env/默认配置，也可改成 "auto"、"cuda"、"mps"、"cpu"。
THRESHOLD = 0.6


def main():
    # 在这里打断点，Step Into 可进入模型加载与推理；result 是完整决策结果。
    result = run_local_test(
        "video",
        QUESTION,
        CANDIDATES,
        VIDEO_PATH,
        project_root=PROJECT_ROOT,
        device=DEVICE,
        threshold=THRESHOLD,
    )
    return result


if __name__ == "__main__":
    main()
