"""直接运行，或修改下方 IMAGE_PATH / QUESTION / CANDIDATES。"""

from pathlib import Path

from raya_maas.local_test import run_local_test

IMAGE_PATH = Path(__file__).resolve().parents[1] / "tests/fixtures/m3bench_living.jpg"
QUESTION = "What object is the person on the right holding?"
CANDIDATES = ["A basketball", "A laptop", "A cup", "A book"]


if __name__ == "__main__":
    run_local_test("image", QUESTION, CANDIDATES, IMAGE_PATH)
