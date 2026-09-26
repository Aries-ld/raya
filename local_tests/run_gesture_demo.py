"""右键 Run 启动摄像头演示页；MaaS 需先运行。打开打印的本机网址。"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from raya_maas.demo_server import serve  # noqa: E402

if __name__ == "__main__":
    serve()
