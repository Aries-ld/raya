#!/usr/bin/env python3
"""GPU 健康分维护 Skill：检测空闲 GPU，自动准备模型和数据，每张卡独立跑真实训练任务。

用法：
    python start.py start                    # 自动检测空闲卡并启动
    python start.py start --gpus 4,5,6,7     # 指定卡
    python start.py status                   # 查看状态
    python start.py stop                     # 停止

    # 自定义配置（不同容器路径不同）
    python start.py start --project-dir /path/to/raya --model-id Qwen/Qwen3.5-2B

设计原则：
    - 所有路径可配置，提供自动检测但不硬编码
    - 模型和数据集自动下载（hf-mirror 可达的自动下，拿不到的跳过）
    - 每张空闲卡独立跑训练（不依赖多卡 DDP，异构卡兼容）
    - batch_size 按卡的显存自适应
    - supervisor 保证训练永不停止，除非显式调用 stop
"""
from __future__ import annotations
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

STATE_DIR = Path("/tmp/raya_keepalive")
STATE_DIR.mkdir(exist_ok=True)

# 默认模型和数据配置（可在不同环境覆盖）
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"
DEFAULT_DATASETS = {
    "race": {"limit": 20000, "modality": "text"},
    "koniq": {"limit": None, "modality": "image"},
    "gqa": {"limit": 50000, "modality": "image"},
    "nextqa": {"limit": None, "modality": "video"},
}


def get_gpu_status() -> list[dict]:
    """返回每张卡的 index, memory_used_mb, total_mb, utilization, name"""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu,name", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    gpus = []
    for line in result.stdout.strip().split("\n"):
        parts = line.split(",")
        gpus.append({
            "index": int(parts[0].strip()),
            "memory_mb": int(parts[1].strip()),
            "total_mb": int(parts[2].strip()),
            "utilization": int(parts[3].strip()),
            "name": parts[4].strip(),
        })
    return gpus


def find_idle_gpus(min_free_mb: int = 80000) -> list[dict]:
    """找显存占用低于阈值的卡"""
    gpus = get_gpu_status()
    return [g for g in gpus if g["memory_mb"] < min_free_mb]


def batch_size_for_gpu(total_mb: int, gpu_name: str = "") -> int:
    """按 GPU 型号和显存自适应 batch_size。

    不同型号的卡算力差异大，同样显存下 H800 可以开更大的 batch：
    - H800: 80GB HBM3，算力最强
    - H20: 96GB HBM3，算力较强
    - Pro 5000-72G: 72GB，算力中等
    - A10/V100: 40-48GB，算力一般
    - T4: 16GB，算力较弱
    """
    name_lower = gpu_name.lower()

    if "h800" in name_lower or "h100" in name_lower:
        return 8 if total_mb >= 80000 else 4
    elif "h20" in name_lower:
        return 4 if total_mb >= 80000 else 2
    elif "pro" in name_lower and "5000" in name_lower:  # Pro 5000-72G
        return 4 if total_mb >= 72000 else 2
    elif total_mb >= 80000:
        return 4
    elif total_mb >= 40000:
        return 2
    elif total_mb >= 20000:
        return 1
    else:
        return 1


def state_file(gpu_index: int) -> Path:
    return STATE_DIR / f"gpu_{gpu_index}.json"


def is_running(gpu_index: int) -> dict | None:
    sf = state_file(gpu_index)
    if not sf.exists():
        return None
    state = json.loads(sf.read_text())
    pid = state.get("pid")
    if pid and Path(f"/proc/{pid}").exists():
        return state
    return None


def detect_project_dir() -> Path:
    """自动检测项目目录：优先当前目录，其次常见路径"""
    candidates = [
        Path.cwd() / "raya",
        Path.cwd(),
        Path("/mnt/tidalfs-hssh01/dataset/linzichen/lindong/raya"),
        Path.home() / "raya",
    ]
    for p in candidates:
        if (p / "raya" / "data" / "schema.py").exists() or (p / "scripts" / "m4" / "train_full_ft.py").exists():
            return p
    return Path.cwd()


def ensure_model_and_data(project_dir: Path, model_id: str) -> None:
    """确保模型和数据集存在，不存在则自动下载（hf-mirror 可达的自动下，拿不到的跳过）"""
    venv_python = project_dir / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = Path(sys.executable)

    # 检查模型是否已缓存
    model_cache = Path.home() / ".cache" / "huggingface" / "hub"
    model_cached = any(model_cache.glob(f"models--{model_id.replace('/', '--')}*"))
    if not model_cached:
        print(f"下载模型 {model_id}...")
        subprocess.run(
            [str(venv_python), "-c", f"from transformers import AutoModelForImageTextToText; AutoModelForImageTextToText.from_pretrained('{model_id}')"],
            env={**os.environ, "HF_ENDPOINT": "https://hf-mirror.com"},
            check=True,
        )

    # 检查数据集
    data_dir = project_dir / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    prepare_script = project_dir / "scripts" / "m3" / "prepare_dataset.py"

    for name, cfg in DEFAULT_DATASETS.items():
        out_file = data_dir / f"{name}_train.jsonl"
        if out_file.exists():
            continue
        if not prepare_script.exists():
            print(f"数据准备脚本不存在: {prepare_script}，跳过 {name}")
            continue
        print(f"准备数据集 {name}...")
        try:
            subprocess.run(
                [str(venv_python), str(prepare_script), "--adapter", name, "--split", "train",
                 *(["--limit", str(cfg["limit"])] if cfg["limit"] else []),
                 "--cache-dir", str(project_dir / "data" / "hf_cache"),
                 "--media-dir", "/data/temp/raya_scratch/data/media" if cfg["modality"] != "text" else str(data_dir),
                 "--out", str(out_file)],
                env={**os.environ, "HF_ENDPOINT": "https://hf-mirror.com"},
                check=True, timeout=600,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"数据集 {name} 准备失败（可能外网不可达），跳过: {e}")


def start_gpu(
    gpu: dict,
    train_script: str,
    model_dir: str,
    data_files: list[str],
    media_root: str,
    venv_python: str,
    project_dir: Path,
) -> None:
    """在单张卡上启动训练进程 + supervisor"""
    gpu_idx = gpu["index"]

    if is_running(gpu_idx):
        print(f"GPU{gpu_idx}: 已在运行，跳过")
        return

    batch_size = batch_size_for_gpu(gpu["total_mb"], gpu["name"])
    log_file = project_dir / "logs" / f"keepalive_gpu{gpu_idx}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    data_args = ", ".join(f'"{f}"' for f in data_files)
    supervisor_code = f"""
import subprocess, sys, time, os
cmd = [
    "{venv_python}", "{train_script}",
    "--model-dir", "{model_dir}",
    "--train-data", {data_args},
    "--media-root", "{media_root}",
    "--max-length", "4096",
    "--batch-size", "{batch_size}",
    "--epochs", "999",
    "--lr", "2e-5",
    "--output-dir", "{project_dir}/checkpoints/keepalive_gpu{gpu_idx}",
]
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "{gpu_idx}"
env["HF_ENDPOINT"] = "https://hf-mirror.com"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
while True:
    proc = subprocess.run(cmd, env=env, cwd="{project_dir}",
                         stdout=open("{log_file}", "a"), stderr=subprocess.STDOUT)
    print(f"training exited with code {{proc.returncode}}, restarting in 10s...", flush=True)
    time.sleep(10)
"""

    supervisor_file = STATE_DIR / f"supervisor_gpu{gpu_idx}.py"
    supervisor_file.write_text(supervisor_code)

    process = subprocess.Popen(
        [sys.executable, str(supervisor_file)],
        stdout=open(log_file, "a"),
        stderr=subprocess.STDOUT,
    )

    state_file(gpu_idx).write_text(json.dumps({
        "pid": process.pid,
        "gpu": gpu_idx,
        "gpu_name": gpu["name"],
        "batch_size": batch_size,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "log": str(log_file),
    }))
    print(f"GPU{gpu_idx} ({gpu['name']}): 已启动 (pid={process.pid}, batch_size={batch_size}, 显存={gpu['total_mb']}MB)")


def start(
    gpu_indices: list[int] | None = None,
    project_dir: Path | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    train_script: str | None = None,
    model_dir: str | None = None,
    data_files: list[str] | None = None,
    media_root: str | None = None,
    venv_python: str | None = None,
) -> None:
    project_dir = project_dir or detect_project_dir()
    print(f"项目目录: {project_dir}")

    # 自动准备模型和数据
    ensure_model_and_data(project_dir, model_id)

    train_script = train_script or str(project_dir / "scripts" / "m4" / "train_full_ft.py")
    model_dir = model_dir or str(project_dir / "checkpoints" / "full_ft_v1")
    data_files = data_files or [
        str(project_dir / "data" / "processed" / f"{name}_train.jsonl")
        for name in DEFAULT_DATASETS
    ]
    media_root = media_root or "/data/temp/raya_scratch"
    venv_python = venv_python or str(project_dir / ".venv" / "bin" / "python")

    # 检查数据文件（跳过不存在的）
    existing_data = [f for f in data_files if Path(f).exists()]
    if not existing_data:
        # fallback：如果拉取不到任何数据集，用 skill 自带的文本训练集（纯文本，不需要媒体文件）
        fallback = Path(__file__).parent / "text_fallback.jsonl"
        if fallback.exists():
            print(f"拉取不到数据集，使用内置文本 fallback 训练集: {fallback}")
            existing_data = [str(fallback)]
        else:
            print("没有可用的训练数据，请先准备数据")
            sys.exit(1)
    if len(existing_data) < len(data_files):
        skipped = set(data_files) - set(existing_data)
        if len(existing_data) == 1 and "text_fallback" in existing_data[0]:
            pass  # fallback 情况，不打印跳过信息
        else:
            print(f"跳过不可用的数据文件: {skipped}")

    if not Path(model_dir).exists():
        print(f"模型不存在: {model_dir}，需要先完成一轮训练")
        sys.exit(1)

    if gpu_indices is not None:
        all_gpus = get_gpu_status()
        gpus = [g for g in all_gpus if g["index"] in gpu_indices]
    else:
        gpus = find_idle_gpus()

    if not gpus:
        print("没有空闲 GPU")
        return

    print(f"发现 {len(gpus)} 张空闲卡，开始启动训练...")
    for gpu in gpus:
        start_gpu(gpu, train_script, model_dir, existing_data, media_root, venv_python, project_dir)


def stop() -> None:
    """停止所有正在运行的训练"""
    stopped = 0
    for sf in STATE_DIR.glob("gpu_*.json"):
        state = json.loads(sf.read_text())
        pid = state.get("pid")
        if pid and Path(f"/proc/{pid}").exists():
            print(f"停止 GPU{state['gpu']} (pid={pid})")
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
            if Path(f"/proc/{pid}").exists():
                os.kill(pid, signal.SIGKILL)
            stopped += 1
        sf.unlink(missing_ok=True)

    for f in STATE_DIR.glob("supervisor_gpu*.py"):
        f.unlink(missing_ok=True)

    print(f"已停止 {stopped} 个训练进程")


def status() -> None:
    gpus = get_gpu_status()
    running = []
    for sf in STATE_DIR.glob("gpu_*.json"):
        state = json.loads(sf.read_text())
        pid = state.get("pid")
        if pid and Path(f"/proc/{pid}").exists():
            running.append(state)

    if not running:
        print("没有正在运行的训练")
    else:
        print(f"正在运行的训练 ({len(running)} 个):")
        for state in running:
            print(f"  GPU{state['gpu']} ({state.get('gpu_name', 'unknown')}): pid={state['pid']}, batch_size={state['batch_size']}, 启动于 {state['started_at']}")

    print("\n所有 GPU 状态:")
    for g in gpus:
        marker = " ← 训练中" if any(r["gpu"] == g["index"] for r in running) else ""
        print(f"  GPU{g['index']} ({g['name']}): {g['memory_mb']}/{g['total_mb']}MB used, {g['utilization']}% util{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU 健康分维护：每张空闲卡独立跑训练，supervisor 保证永不停止")
    parser.add_argument("action", choices=["start", "stop", "status"])
    parser.add_argument("--gpus", type=str, default=None, help="指定 GPU，如 '4,5,6,7'")
    parser.add_argument("--project-dir", type=str, default=None, help="项目目录（默认自动检测）")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="模型 ID（默认 Qwen/Qwen3.5-2B）")
    parser.add_argument("--train-script", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--data-files", type=str, nargs="+", default=None)
    parser.add_argument("--media-root", type=str, default=None)
    parser.add_argument("--venv-python", type=str, default=None)
    args = parser.parse_args()

    if args.action == "start":
        gpu_indices = [int(x) for x in args.gpus.split(",")] if args.gpus else None
        start(
            gpu_indices=gpu_indices,
            project_dir=Path(args.project_dir) if args.project_dir else None,
            model_id=args.model_id,
            train_script=args.train_script,
            model_dir=args.model_dir,
            data_files=args.data_files,
            media_root=args.media_root,
            venv_python=args.venv_python,
        )
    elif args.action == "stop":
        stop()
    elif args.action == "status":
        status()


if __name__ == "__main__":
    main()
