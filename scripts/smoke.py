"""Real-model, real-HTTP smoke tests via the official OpenAI Python client."""

import argparse
import base64
import json
import os
import time
from pathlib import Path

from openai import OpenAI


def data_url(path):
    kind = "image/jpeg" if path.suffix == ".jpg" else "video/mp4"
    return f"data:{kind};base64," + base64.b64encode(path.read_bytes()).decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke.json"))
    args = parser.parse_args()
    client = OpenAI(
        base_url=args.base_url,
        api_key=os.environ.get("RAYA_TEST_API_KEY", "local"),
        timeout=180,
        max_retries=0,
    )
    cases = [
        (
            "text_zh",
            "用户说：帮我把明天下午3点的会议改到4点。这个请求的意图是什么？",
            ["修改会议时间", "创建新会议", "取消会议", "查询会议详情"],
            None,
            "A",
        ),
        (
            "text_en",
            "The user says: Please cancel my reservation. What is the intent?",
            ["Create a reservation", "Cancel a reservation", "Check the weather"],
            None,
            "B",
        ),
        (
            "image",
            "What object is the person on the right holding?",
            ["A basketball", "A laptop", "A cup", "A book"],
            "m3bench_living.jpg",
            "A",
        ),
        (
            "video_kitchen",
            "Where does this video take place?",
            ["A kitchen", "A beach", "An office", "A street"],
            "m3bench_kitchen.mp4",
            None,
        ),
        (
            "video_living",
            "Where does this video take place?",
            ["An indoor room", "A beach", "A forest", "A street"],
            "m3bench_living.mp4",
            None,
        ),
    ]
    results = []
    for name, question, candidates, filename, expected in cases:
        content = [{"type": "text", "text": question}]
        if filename:
            kind = "image" if filename.endswith(".jpg") else "video"
            content.insert(
                0,
                {"type": f"{kind}_url", f"{kind}_url": {"url": data_url(args.fixtures / filename)}},
            )
        start = time.perf_counter()
        response = client.chat.completions.create(
            model="raya-decision-v1",
            messages=[{"role": "user", "content": content}],
            extra_body={"candidates": candidates},
        )
        decision = response.decision
        assert response.choices[0].message.content == decision["label"]
        assert abs(sum(decision["probabilities"].values()) - 1) < 1e-5
        assert decision["selected"] in candidates
        assert 0 <= decision["confidence"] <= 1
        if filename and filename.endswith(".mp4"):
            assert decision["video"][0]["sampled_frames"] == 8
        row = {
            "case": name,
            "wall_ms": round((time.perf_counter() - start) * 1000, 3),
            "expected": expected,
            "matches_expected": None if expected is None else decision["label"] == expected,
            "decision": decision,
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        # Persist partial progress if a subsequent modality fails.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    streamed = list(
        client.chat.completions.create(
            model="raya-decision-v1",
            messages=[{"role": "user", "content": "What is 2 + 2?\n\nA. 4\nB. 5"}],
            stream=True,
            stream_options={"include_usage": True},
            response_format={"type": "json_object"},
        )
    )
    combined = "".join(c.choices[0].delta.content or "" for c in streamed if c.choices)
    assert json.loads(combined)["label"] in ("A", "B")
    assert streamed[-1].usage.completion_tokens == 0
    print(f"PASS: {len(results)} multimodal requests + SDK JSON/SSE/usage")


if __name__ == "__main__":
    main()
