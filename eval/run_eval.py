#!/usr/bin/env python3
"""Raya 评测脚本：在 held-out benchmark 上评估决策准确率和置信度校准。

每个评测集产出：
- accuracy（准确率）
- ECE（Expected Calibration Error，置信度校准）
- latency（p50/p99 延迟）
- 逐样本预测明细（jsonl）

用法:
    python eval/run_eval.py --dataset mmlu --limit 200
    python eval/run_eval.py --dataset banking77 --limit 100
"""
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL_ID = "yuyu199741/raya-decision-v1"
RESULTS_DIR = Path(__file__).parent / "results"


def load_model(model_dir: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16).to(device)
    model.eval()
    return model, tokenizer


def resolve_label_token_ids(tokenizer, labels: list[str]) -> dict[str, int]:
    """给每个标签分配唯一 token id。"""
    label_to_token_id: dict[str, int] = {}
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
    total = len(predictions)
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        acc = sum(int(c) for _, c in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(acc - avg_conf)
    return ece


def evaluate_text(
    model,
    tokenizer,
    samples: list[dict],
    *,
    seed: int = 0,
) -> dict:
    """在文本样本上评估。每个 sample 是 {question, candidates, answer_index}。"""
    rng = random.Random(seed)
    labels_all = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    predictions = []
    latencies = []

    for sample in samples:
        candidates = sample["candidates"]
        labels = labels_all[: len(candidates)]

        # 随机打乱标签顺序
        shuffled = list(zip(labels, candidates))
        rng.shuffle(shuffled)
        labels_shuffled = [l for l, _ in shuffled]
        candidates_shuffled = [c for _, c in shuffled]

        prompt = f"{sample['question']}\n\n" + "\n".join(
            f"{l}. {c}" for l, c in zip(labels_shuffled, candidates_shuffled)
        ) + "\n\nAnswer with exactly one label from the options above."

        label_token_ids = resolve_label_token_ids(tokenizer, labels_shuffled)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        latency = time.perf_counter() - t0

        last_logits = outputs.logits[0, -1, :]
        dist = masked_label_distribution(last_logits, label_token_ids)

        pred_label = max(dist, key=dist.get)
        pred_index = labels_shuffled.index(pred_label)
        pred_option = candidates_shuffled[pred_index]

        is_correct = pred_option == sample["candidates"][sample["answer_index"]]
        confidence = dist[pred_label]
        predictions.append((confidence, is_correct))
        latencies.append(latency)

    correct = sum(int(c) for _, c in predictions)
    accuracy = correct / len(predictions)
    ece = compute_ece(predictions)
    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2]
    p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(predictions),
        "ece": ece,
        "latency_p50_ms": round(p50 * 1000, 1),
        "latency_p99_ms": round(p99 * 1000, 1),
    }


# ── 数据集加载器 ──────────────────────────────────────────────────────────────

def load_mmlu(limit: int | None = None) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test", streaming=True)
    samples = []
    for item in ds:
        samples.append({
            "question": item["question"],
            "candidates": item["choices"],
            "answer_index": item["answer"],
        })
        if limit and len(samples) >= limit:
            break
    return samples


def load_supergpqa(limit: int | None = None) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("m-a-p/SuperGPQA", split="train", streaming=True)
    samples = []
    for item in ds:
        options = item["options"]
        answer_letter = item["answer_letter"]
        answer_index = ord(answer_letter) - ord("A")
        if answer_index >= len(options):
            continue
        samples.append({
            "question": item["question"],
            "candidates": options,
            "answer_index": answer_index,
        })
        if limit and len(samples) >= limit:
            break
    return samples


def load_boolq(limit: int | None = None) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("google/boolq", split="validation", streaming=True)
    samples = []
    for item in ds:
        samples.append({
            "question": f"{item['passage']}\n\n{item['question']}",
            "candidates": ["yes", "no"],
            "answer_index": 0 if item["answer"] else 1,
        })
        if limit and len(samples) >= limit:
            break
    return samples


def load_ag_news(limit: int | None = None) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news", split="test", streaming=True)
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    samples = []
    for item in ds:
        samples.append({
            "question": f"Classify the topic of this news article:\n\n{item['text']}",
            "candidates": labels,
            "answer_index": item["label"],
        })
        if limit and len(samples) >= limit:
            break
    return samples


def load_sst5(limit: int | None = None) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("SetFit/sst5", split="test", streaming=True)
    labels = ["very negative", "negative", "neutral", "positive", "very positive"]
    samples = []
    for item in ds:
        samples.append({
            "question": f"Rate the sentiment of this text:\n\n{item['text']}",
            "candidates": labels,
            "answer_index": item["label"],
        })
        if limit and len(samples) >= limit:
            break
    return samples


DATASETS = {
    "mmlu": load_mmlu,
    "supergpqa": load_supergpqa,
    "boolq": load_boolq,
    "ag_news": load_ag_news,
    "sst5": load_sst5,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--model-dir", default=MODEL_ID)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logger.info("加载模型 {}", args.model_dir)
    model, tokenizer = load_model(args.model_dir, args.device)

    logger.info("加载数据集 {}", args.dataset)
    samples = DATASETS[args.dataset](limit=args.limit)
    logger.info("加载 {} 条样本", len(samples))

    result = evaluate_text(model, tokenizer, samples, seed=args.seed)
    result["dataset"] = args.dataset
    result["model"] = args.model_dir

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info("结果写入 {}", out_path)


if __name__ == "__main__":
    main()
