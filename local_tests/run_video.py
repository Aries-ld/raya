"""直接编辑 video.json 的 state / questions，右键 Run 或 Debug。"""

import sys
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parent))

from raya_maas.local_test import load_request_file, run_local_decision  # noqa: E402


def main():
    request = load_request_file(DIRECTORY / "video.json")  # 完整决策请求，可在此查看/修改。
    result = run_local_decision(request, DIRECTORY.parent)
    print(result)
    return result  # 响应只有 model / answers / usage，与 HTTP 接口相同。


if __name__ == "__main__":
    main()
