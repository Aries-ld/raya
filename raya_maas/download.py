import json
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import Settings

MODEL_REPO = "yuyu199741/raya-decision-v1"
MODEL_REVISION = "72b4c1e433b252e08a915e8bb7cde425cc963f5c"
PROCESSOR_REPO = "Qwen/Qwen3.5-2B"
PROCESSOR_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


def main():
    settings = Settings()
    manifest_path = Path(settings.model_path).parent / "manifest.json"
    revision = PROCESSOR_REVISION
    snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_dir=settings.model_path)
    snapshot_download(
        PROCESSOR_REPO,
        revision=revision,
        local_dir=settings.processor_path,
        allow_patterns=["*.json", "*.jinja", "*.txt"],
        ignore_patterns=["model*"],
    )
    manifest_path.write_text(
        json.dumps(
            {
                "model_repo": MODEL_REPO,
                "model_revision": MODEL_REVISION,
                "processor_repo": PROCESSOR_REPO,
                "processor_revision": revision,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Model and processor ready; revisions recorded in {manifest_path}")


if __name__ == "__main__":
    main()
