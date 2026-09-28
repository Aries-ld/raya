#!/usr/bin/env python3
"""GPU 健康分维护 Skill：检测空闲 GPU，每张卡独立跑真实训练任务，supervisor 保证永不停止。

用法：
    python start.py          # 启动（自动检测空闲卡）
    python start.py --gpus 4,5,6,7   # 指定卡
    python stop.py           # 停止所有
    python status.py         # 查看状态

设计原则：
    - 每张空闲卡独立跑一个训练进程（不依赖多卡 DDP，避免异构卡兼容问题）
    - batch_size 按卡的显存自动调整
    - supervisor 监控训练进程，退出后自动重启（永不停止）
    - 只有用户显式调用 stop.py 才会停止
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

# 默认配置（Raya 项目）
DEFAULT_PROJECT_DIR = Path("/mnt/tidalfs-hssh01/dataset/linzichen/lindong/raya")
DEFAULT_VENV = DEFAULT_PROJECT_DIR / ".venv" / "bin" / "python"
DEFAULT_TRAIN_SCRIPT = DEFAULT_PROJECT_DIR / "scripts" / "m4" / "train_full_ft.py"
DEFAULT_MODEL_DIR = DEFAULT_PROJECT_DIR / "checkpoints" / "full_ft_v1"
DEFAULT_DATA_FILES = [
    DEFAULT_PROJECT_DIR / "data" / "processed" / "race_train.jsonl",
    DEFAULT_PROJECT_DIR / "data" / "processed" / "koniq_train.jsonl",
    DEFAULT_PROJECT_DIR / "data" / "processed" / "gqa_train.jsonl",
    DEFAULT_PROJECT_DIR / "data" / "processed" / "nextqa_train.jsonl",
]
DEFAULT_MEDIA_ROOT = "/data/temp/raya_scratch"


def get_gpu_status() -> list[dict]:
    """返回每张卡的 index, memory_used_mb, total_mb, utilization"""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
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
        })
    return gpus


def find_idle_gpus(min_free_mb: int = 80000) -> list[dict]:
    """找显存占用低于阈值的卡，返回完整信息（含总显存）"""
    gpus = get_gpu_status()
    return [g for g in gpus if g["memory_mb"] < min_free_mb]


def batch_size_for_gpu(total_mb: int) -> int:
    """按显存大小自适应 batch_size"""
    if total_mb >= 80000:  # 80GB+
        return 4
    elif total_mb >= 40000:  # 40GB+
        return 2
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

    batch_size = batch_size_for_gpu(gpu["total_mb"])
    log_file = project_dir / "logs" / f"keepalive_gpu{gpu_idx}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # supervisor 脚本：训练退出后自动重启
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
        "batch_size": batch_size,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "log": str(log_file),
    }))
    print(f"GPU{gpu_idx}: 已启动 (pid={process.pid}, batch_size={batch_size}, 显存={gpu['total_mb']}MB)")


def start(
    gpu_indices: list[int] | None = None,
    train_script: str | None = None,
    model_dir: str | None = None,
    data_files: list[str] | None = None,
    media_root: str | None = None,
    venv_python: str | None = None,
    project_dir: Path | None = None,
) -> None:
    train_script = train_script or str(DEFAULT_TRAIN_SCRIPT)
    model_dir = model_dir or str(DEFAULT_MODEL_DIR)
    data_files = data_files or [str(f) for f in DEFAULT_DATA_FILES]
    media_root = media_root or DEFAULT_MEDIA_ROOT
    venv_python = venv_python or str(DEFAULT_VENV)
    project_dir = project_dir or DEFAULT_PROJECT_DIR

    # 检查数据文件
    missing = [f for f in data_files if not Path(f).exists()]
    if missing:
        print(f"缺少数据文件: {missing}")
        sys.exit(1)
    if not Path(model_dir).exists():
        print(f"模型不存在: {model_dir}")
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
        start_gpu(gpu, train_script, model_dir, data_files, media_root, venv_python, project_dir)


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

    # 清理 supervisor 脚本
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
            print(f"  GPU{state['gpu']}: pid={state['pid']}, batch_size={state['batch_size']}, 启动于 {state['started_at']}")

    print("\n所有 GPU 状态:")
    for g in gpus:
        marker = " ← 训练中" if any(r["gpu"] == g["index"] for r in running) else ""
        print(f"  GPU{g['index']}: {g['memory_mb']}/{g['total_mb']}MB used, {g['utilization']}% util{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU 健康分维护：每张空闲卡独立跑训练，supervisor 保证永不停止")
    parser.add_argument("action", choices=["start", "stop", "status"])
    parser.add_argument("--gpus", type=str, default=None, help="指定 GPU，如 '4,5,6,7'")
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
