"""右键 Run / Debug 即可。直接编辑同目录 text.txt；素材也在本目录。"""

import sys
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parent))

from raya_maas.local_test import run_case_file  # noqa: E402


def main():
    result = run_case_file("text", DIRECTORY)
    return result  # 在此打断点，查看标签、选项、概率、置信度及耗时。


if __name__ == "__main__":
    main()
