"""直接运行，或修改下方 VIDEO_PATH / QUESTION / CANDIDATES。"""

from pathlib import Path

from raya_maas.local_test import run_local_test

VIDEO_PATH = Path(__file__).resolve().parents[1] / "tests/fixtures/m3bench_kitchen.mp4"
QUESTION = "Where does this video take place?"
CANDIDATES = ["A kitchen", "A beach", "An office", "A street"]


if __name__ == "__main__":
    run_local_test("video", QUESTION, CANDIDATES, VIDEO_PATH)
