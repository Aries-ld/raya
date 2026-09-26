"""Real-model HTTP checks using the same decision JSON as the offline IDE tests."""

import argparse
import json
from pathlib import Path

import httpx

from raya_maas.config import Settings
from raya_maas.decisions import SystemOneResponse
from raya_maas.local_test import load_request_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", type=Path, default=Path("local_tests"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/systemone-http.json"))
    args = parser.parse_args()
    settings = Settings()
    headers = {"Authorization": f"Bearer {settings.keys[0]}"} if settings.keys else {}
    results = {}
    with httpx.Client(base_url=args.base_url, headers=headers, timeout=180) as client:
        for modality, question_id, expected in [
            ("text", "intent", "reset_password"),
            ("image", "held_object", "basketball"),
            ("video", "food_in_box", "fries"),
        ]:
            request = load_request_file(args.cases / f"{modality}.json", settings)
            response = client.post("/v1/systemone", json=request.model_dump())
            response.raise_for_status()
            result = SystemOneResponse.model_validate(response.json()).model_dump()
            assert result["answers"][question_id]["choice"] == expected
            for answer in result["answers"].values():
                if "probabilities" in answer:
                    assert abs(sum(answer["probabilities"].values()) - 1) < 1e-5
            results[modality] = result
            print(modality, json.dumps(result, ensure_ascii=False), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
        # Incomplete calls must be rejected, not guessed from media or context alone.
        bad = client.post("/v1/systemone", json={"model": settings.model_name, "state": "hello"})
        assert bad.status_code == 422
        old = client.post("/v1/chat/completions", json={})
        assert old.status_code == 404
    print("PASS: native text/image/video decisions, all three question types, validation")


if __name__ == "__main__":
    main()
