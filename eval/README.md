# Evaluation

All benchmarks are **held-out**: no evaluation sample was seen during training. Every question was rendered with a fixed template, candidates were shuffled with a fixed seed, and each decision was made in a single forward pass with masked logit projection over candidate label tokens.

## Setup

- **Model**: [`yuyu199741/raya-decision-v1`](https://huggingface.co/yuyu199741/raya-decision-v1) (Qwen3.5-2B, 2.2B params, BF16)
- **Hardware**: NVIDIA H20 96GB
- **Evaluation date**: 2026-09-26
- **Inference**: single forward pass, no text generation

## Metrics

- **Accuracy**: fraction of samples where the predicted candidate matches the gold answer
- **ECE** (Expected Calibration Error): weighted average gap between predicted confidence and actual accuracy across 10 confidence buckets. Lower is better. Target: ≤ 0.08
- **Latency**: wall-clock time for a single forward pass (p50 / p99)

## Results

### Text

| Benchmark | Task | Samples | Accuracy | ECE | p50 | p99 |
|---|---|---|---|---|---|---|
| AG News | Topic classification (4-way) | 200 | **78.5%** | 0.082 | 45ms | 114ms |
| BoolQ | Yes/no reading comprehension | 200 | **74.0%** | 0.169 | 86ms | 883ms |
| SST-5 | Sentiment analysis (5-way) | 200 | 39.5% | 0.210 | 45ms | 117ms |
| MMLU | Graduate-level knowledge MC | 50 | 20.0% | 0.367 | 86ms | 2095ms |
| SuperGPQA | Graduate-level reasoning | 200 | 9.5% | 0.250 | 87ms | 112ms |

### Image

| Benchmark | Task | Samples | Accuracy | ECE | p50 | p99 |
|---|---|---|---|---|---|---|
| GQA val | Visual QA (choice + yes/no) | 20 | **100.0%** | 0.015 | 104ms | 1565ms |
| KonIQ val | Image quality scoring (5-way) | 20 | **90.0%** | 0.194 | 59ms | 1572ms |
| POPE | Object existence (yes/no) | 200 | **88.5%** | **0.061** | 60ms | 964ms |

### Video

| Benchmark | Task | Samples | Accuracy | ECE | p50 | p99 |
|---|---|---|---|---|---|---|
| NExT-QA val | Video content QA (5-way) | 20 | **90.0%** | 0.150 | 155ms | 1403ms |

## Notes

- **GQA val 100%**: small sample (20), likely reflects the relative simplicity of balanced GQA questions rather than a claim of perfect image understanding.
- **MMLU / SuperGPQA low scores**: these are pure knowledge/reasoning questions with no contextual passage. Raya was trained on decision tasks with context (reading comprehension, image QA, video QA), not on pure factual recall. A 2.2B model has limited parametric knowledge for graduate-level questions.
- **SST-5 low score**: 5-way sentiment is a fine-grained ordinal task; the model was not specifically trained on sentiment. Score-type questions benefit from soft-label training data that was not available for this domain.
- **POPE ECE 0.061**: best calibration across all benchmarks — the yes/no format with clear visual grounding produces the most reliable confidence estimates.
- **BoolQ ECE 0.169**: the passage-based yes/no format produces reasonable accuracy but overconfident predictions, likely because passage comprehension is harder to calibrate than visual grounding.

## Training data

| Dataset | Modality | Training samples | License |
|---|---|---|---|
| RACE | Text | 40,000 | research-only |
| GQA | Image | 200,000 | research-only |
| KonIQ-10k | Image | 7,058 | research-only |
| NExT-QA | Video | 7,209 | research-only |

## Reproduce

```bash
# Text benchmarks
python eval/run_eval.py --dataset ag_news --limit 200
python eval/run_eval.py --dataset boolq --limit 200
python eval/run_eval.py --dataset sst5 --limit 200
python eval/run_eval.py --dataset mmlu --limit 50
python eval/run_eval.py --dataset supergpqa --limit 200

# Image benchmarks
python eval/run_pope.py --limit 200
python eval/run_image.py --dataset gqa
python eval/run_image.py --dataset koniq

# Video benchmark
python eval/run_nextqa.py
```
