#!/usr/bin/env python3
"""GQA val + KonIQ val 图片评测（held-out）。"""
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from PIL import Image

MODEL_ID = "yuyu199741/raya-decision-v1"
RESULTS_DIR = Path(__file__).parent / "results"
MEDIA_ROOT = Path("/data/temp/raya_scratch")
EYEBALL_DIR = Path("/mnt/tidalfs-hssh01/dataset/linzichen/lindong/raya/data/eyeball")


def resolve_label_token_ids(tokenizer, labels: list[str]) -> dict[str, int]:
    label_to_token_id = {}
    for label in labels:
        token_id = None
        for candidate in (f" {label}", label):
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) == 1:
                token_id = ids[0]
                break
        if token_id is None:
            raise ValueError(f"标签 '{label}' 无法编码为单 token")
        label_to_token_id[label] = token_id
    return label_to_token_id


def masked_label_distribution(last_logits: torch.Tensor, label_to_token_id: dict[str, int]) -> dict[str, float]:
    labels = list(label_to_token_id)
    token_ids = torch.tensor([label_to_token_id[l] for l in labels], device=last_logits.device)
    selected = last_logits.index_select(0, token_ids)
    probs = torch.softmax(selected, dim=-1)
    return {label: probs[i].item() for i, label in enumerate(labels)}


def compute_ece(predictions: list[tuple[float, bool]], n_bins: int = 10) -> float:
    bins = [[] for _ in range(n_bins)]
    for conf, correct in predictions:
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, correct))
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        acc = sum(int(c) for _, c in bucket) / len(bucket)
        ece += (len(bucket) / len(predictions)) * abs(acc - avg_conf)
    return ece


def evaluate_image_dataset(model, tokenizer, processor, samples, media_root, device, seed):
    rng = random.Random(seed)
    labels_all = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    predictions = []
    latencies = []

    for sample in samples:
        image_path = media_root / sample["media_path"]
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        question = sample["question"]["text"]
        options = [opt["text"] for opt in sample["options"]]
        answer_id = sample["answer"]

        labels = list(labels_all[: len(options)])
        rng.shuffle(labels)
        label_to_option_id = {l: opt["id"] for l, opt in zip(labels, sample["options"])}

        text = f"<|vision_start|><|image_pad|><|vision_end|>\n\n{question}\n\n"
        text += "\n".join(f"{l}. {t}" for l, t in zip(labels, options))
        text += "\n\nAnswer with exactly one label from the options above."

        label_token_ids = resolve_label_token_ids(tokenizer, labels)
        enc = processor(text=[text], images=[image], return_tensors="pt").to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(**enc)
        latency = time.perf_counter() - t0

        last_logits = outputs.logits[0, -1, :]
        dist = masked_label_distribution(last_logits, label_token_ids)

        pred_label = max(dist, key=dist.get)
        pred_option_id = label_to_option_id[pred_label]
        is_correct = pred_option_id == answer_id
        confidence = dist[pred_label]
        predictions.append((confidence, is_correct))
        latencies.append(latency)

    correct = sum(int(c) for _, c in predictions)
    accuracy = correct / len(predictions)
    ece = compute_ece(predictions)
    latencies_sorted = sorted(latencies)

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(predictions),
        "ece": ece,
        "latency_p50_ms": round(latencies_sorted[len(latencies_sorted) // 2] * 1000, 1),
        "latency_p99_ms": round(latencies_sorted[int(len(latencies_sorted) * 0.99)] * 1000, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["gqa", "koniq"])
    parser.add_argument("--model-dir", default=MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logger.info("加载模型 {}", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-2B")
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16).to(args.device)
    model.eval()

    # 加载对应的 held-out eyeball 数据
    eyeball_path = EYEBALL_DIR / f"{args.dataset}_val_20.jsonl"
    samples = []
    with open(eyeball_path) as f:
        for line in f:
            samples.append(json.loads(line))
    logger.info("加载 {} 条 {} val 样本", len(samples), args.dataset)

    result = evaluate_image_dataset(model, tokenizer, processor, samples, MEDIA_ROOT, args.device, args.seed)
    result["dataset"] = f"{args.dataset}_val"
    result["model"] = args.model_dir

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.dataset}_val.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info("结果写入 {}", out_path)


if __name__ == "__main__":
    main()
