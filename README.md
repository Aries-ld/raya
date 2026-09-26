<h1 align="center">Raya</h1>
<p align="center"><strong>System 1 decisions over text, images, and video</strong></p>

<div align="center">

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Raya%20v1-FFD21E)](https://huggingface.co/yuyu199741/raya-decision-v1)
[![Base Model](https://img.shields.io/badge/Base%20Model-Qwen3.5--2B-7C3AED)](https://huggingface.co/Qwen/Qwen3.5-2B)
[![API Docs](https://img.shields.io/badge/Docs-API-2ea44f)](docs/api.md)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](pyproject.toml)

**English** | [简体中文](README.zh-CN.md)

[Quick start](#quick-start) · [API guide](docs/api.md) · [Examples](docs/examples/)

</div>

A decision service built on a post-trained Qwen3.5-2B checkpoint. Ask **choice, score, and yes/no questions** over text, images, or short videos. Each question uses one forward pass and returns structured JSON, with no free-form text generation to parse.

## Benchmarks

All results are held-out (never seen during training). Every question is answered with a single forward pass and masked logit projection over candidate label tokens. See [eval/README.md](eval/README.md) for full methodology and per-benchmark details.

| Benchmark | Modality | Task | Accuracy | ECE | p50 Latency |
|---|---|---|---|---|---|
| GQA val | Image | Visual QA | **100.0%** | 0.015 | 104ms |
| KonIQ val | Image | Quality scoring (5-way) | **90.0%** | 0.194 | 59ms |
| NExT-QA val | Video | Video content QA | **90.0%** | 0.150 | 155ms |
| POPE | Image | Object existence (yes/no) | **88.5%** | 0.061 | 60ms |
| AG News | Text | Topic classification (4-way) | **78.5%** | 0.082 | 45ms |
| BoolQ | Text | Yes/no reading comprehension | **74.0%** | 0.169 | 86ms |
| SST-5 | Text | Sentiment (5-way) | 39.5% | 0.210 | 45ms |
| MMLU | Text | Graduate-level knowledge | 20.0% | 0.367 | 86ms |
| SuperGPQA | Text | Graduate-level reasoning | 9.5% | 0.250 | 87ms |

**Key takeaways:**
- Strongest on image and video decision tasks (88–100%)
- Confidence calibration is well-behaved on high-confidence predictions (ECE 0.015–0.061 on POPE/GQA)
- Pure knowledge/reasoning benchmarks (MMLU, SuperGPQA) are not the target use case — Raya is a decision model, not a knowledge retrieval model
- Text classification and reading comprehension tasks are solid (74–78%)

## Quick start

Requires Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/). The same codebase supports CUDA, Apple Silicon MPS, and CPU.

```bash
git clone https://github.com/Aries-ld/raya.git
cd raya

uv sync --frozen --extra dev
uv run raya-download

cp .env.example .env
# Edit .env and replace RAYA_API_KEYS with your own random secret.

uv run raya-serve
```

Do not overwrite an existing `.env`. Clients must use a key accepted by the server. Keep keys, weights, and local artifacts out of Git. Configure the bind address and port with `RAYA_HOST` and `RAYA_PORT`.

Check readiness after startup:

```bash
curl http://127.0.0.1:8000/readyz
# {"status":"ready"}
```

Before initialization completes, readiness checks may return 503 or fail to connect. For remote use, the operator must provide a reachable service URL and API key; `127.0.0.1` always refers to the caller's own machine.

## API example

Set client environment variables. `RAYA_BASE_URL` is the service root URL, without `/v1`.

```bash
export RAYA_BASE_URL='http://127.0.0.1:8000'
export RAYA_API_KEY='replace-with-your-issued-key'

curl --fail-with-body --max-time 150 \
  "$RAYA_BASE_URL/v1/systemone" \
  -H "Authorization: Bearer $RAYA_API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "raya-decision-v1",
    "state": "I forgot my password and cannot sign in.",
    "questions": {
      "intent": {
        "type": "choice",
        "instructions": "What is the main request?",
        "criteria": {
          "reset_password": "Recover or reset a login password",
          "register": "Create a new account",
          "other": "A different request"
        }
      }
    }
  }'
```

Illustrative response; probabilities and token counts below are examples, not a measured guarantee:

```json
{
  "model": "raya-decision-v1",
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "reset_password",
      "probabilities": {"reset_password": 0.98, "register": 0.01, "other": 0.01},
      "confidence": 0.98
    }
  },
  "usage": {"input_tokens": 96, "output_tokens": 0}
}
```

The response uses your option IDs, so no generated prose needs to be parsed. `confidence` is currently the maximum candidate probability; a high value does not guarantee correctness when the candidate set is incomplete.

See the **[API integration guide (Chinese)](docs/api.md)** for authentication, media inputs, all three question types, limits, errors, and retry behavior. A machine-readable [OpenAPI schema](docs/openapi.json) is included.

### Standalone Python client

The example client uses only the Python standard library. Callers do not need model weights or inference dependencies.

```bash
python3 docs/examples/call_raya.py --request docs/examples/text-request.json
python3 docs/examples/call_raya.py --request docs/examples/image-request.json --image ./photo.jpg
python3 docs/examples/call_raya.py --request docs/examples/video-request.json --video ./clip.mp4
```

For media requests, the client converts local files to Base64 data URLs. The HTTP API does not accept local file paths; remote HTTPS media hosts must be allowlisted by the operator.

## Documentation

| Resource | Contents |
| --- | --- |
| [API integration guide (Chinese)](docs/api.md) | Authentication, question types, media, responses, errors, and retries |
| [OpenAPI schema](docs/openapi.json) | Machine-readable request and response definitions |
| [Client examples](docs/examples/) | A standard-library Python client and request templates for all three modalities |
| [Configuration](.env.example) | Device selection, model paths, request limits, and decoding settings |

## How it works

| Capability | Behavior |
| --- | --- |
| Text | Natural-language or structured JSON context |
| Images | Decisions over text context and image inputs |
| Video | 3–20 seconds, Qwen `fps=1` sampling, visual content only |
| `choice` | Select from 2–26 options with a full probability distribution |
| `score` | Return an expected score over 2–10 ordered levels |
| `noul` | Return the probability of yes |
| Multiple questions | 1–16 independent questions per request |
| Devices | Automatic CUDA → MPS → CPU selection and backend fallback |

The decision endpoint is **`POST /v1/systemone`**:

```text
model + state + questions  →  model + answers + usage
```

The API follows the Jev System One request/response style, with images and videos added through `state.media`. Raya uses its own checkpoint, confidence definition, and execution implementation; it does not claim to reproduce Jev's full capabilities, calibration, or throughput. Chat completions, streaming text generation, and tool calling are not provided.

```text
Context + options → multimodal preprocessing → one forward per question → probabilities → JSON
```

## Model artifacts

| Item | Source |
| --- | --- |
| Decision checkpoint | [yuyu199741/raya-decision-v1](https://huggingface.co/yuyu199741/raya-decision-v1) |
| Base model | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) |
| Tokenizer | Loaded from the Raya checkpoint |
| Vision processor | Loaded from Qwen3.5-2B |
| Revision pinning | Repository commits pinned in [download.py](raya_maas/download.py) |

The initial model weights download is approximately 4.4GB. Downloaded revisions are recorded in `models/manifest.json`; serving uses local files only.

## Default limits

| Item | Limit |
| --- | --- |
| Input tokens | 4,096 per question, including context, options, and visual tokens |
| Questions | 1–16 per request |
| Media | Up to 4 items, at most 1 video |
| File size | Up to 64MiB per item |
| HTTP body | Up to 96MiB, including Base64 overhead |
| Video | 3–20 seconds, inclusive |
| Source pixels | Up to 20 million pixels per image or video frame |
| Queue | Up to 8 waiting requests |
| Timeout | 120 seconds by default |

Video sampling follows Qwen's `fps=1` policy: normally at least 4 frames, around 10 for 10 seconds and 20 for 20 seconds. An odd frame count is padded by repeating the last frame during encoding. Sampling is not strictly aligned to integer-second timestamps, and audio is not processed.

Limit violations return explicit errors rather than silent truncation. Questions currently run sequentially; decoded media is reused, but shared-prefix parallel inference and dynamic batching are not implemented. Per-request processing, forward-pass, and queue times are exposed through `X-Raya-*-Ms` response headers.

## Runtime and deployment

- Automatic mode selects CUDA → MPS → CPU; explicitly selecting a device disables device fallback.
- Automatic precision uses CUDA BF16 when supported, otherwise FP16; MPS FP16; CPU FP32.
- A single worker owns the model. An in-flight forward pass may continue after a request times out; retries may repeat computation.
- Per-frame pixel budgets are bounded. Video decoding defaults to 8 CPU threads and selects seeking or sequential decoding based on keyframes.
- Server credentials belong in `.env`; use a gateway for public TLS termination and per-client quotas.

```bash
# CPU container
docker compose up -d --build

# NVIDIA host with drivers and NVIDIA Container Toolkit
docker compose -f compose.yaml -f compose.cuda.yaml up -d --build
```

Containers use a non-root user and a read-only model mount. Use native Python for macOS MPS; Linux containers in Docker Desktop cannot use MPS. Docker builds and CUDA hardware performance have not been verified in the current development environment.

## Local verification

```bash
uv run ruff check raya_maas scripts tests docs/examples
uv run pytest -q

# Real HTTP checks against a running service
python3 docs/examples/call_raya.py --request docs/examples/text-request.json
```

The private `local_tests/` directory is excluded from version control. Public client examples are available in `docs/examples/`; callers supply their own images and videos.

Verification covers the protocol, authentication, input limits, sampling consistency, device fallback, request scheduling, and timing. Small-sample results are not overall accuracy claims or production SLAs; 20-second videos on MPS can still take more than 2 seconds. See the reports for measured results and limitations.

- [Verification record](docs/validation.md)
- [1fps evaluation and latency](docs/fps1-validation.md)
- [Video pipeline audit](docs/video-audit.md)

## Repository layout

```text
raya_maas/        # Inference and HTTP service
scripts/         # Media preparation, checks, benchmarks
tests/           # Automated tests
docs/            # Integration guide and verification records
docs/examples/   # Standalone client and request templates
models/          # Downloaded locally, excluded from Git
artifacts/       # Local outputs, excluded from Git
```

## Contributing

Use [Issues](https://github.com/Aries-ld/raya/issues) for reproducible bug reports, or submit a focused pull request. Include the device, dependency versions, input modality, expected behavior, and actual result. Remove credentials and private media from shared examples.

Run the relevant tests and formatting checks before submitting changes. Keep the API guide, request examples, and OpenAPI definition in sync with protocol changes.

## License

The source code is licensed under the [Apache License 2.0](LICENSE). This code license does not grant rights to model weights, the base model, or external media; those artifacts are governed by their respective publishers' terms.

## References

- [Raya model on Hugging Face](https://huggingface.co/yuyu199741/raya-decision-v1)
- [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- [TypeSafe System One API](https://docs.typesafe.ai/api)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
