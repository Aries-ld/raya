"""直接运行，或修改下方 QUESTION / CANDIDATES；也支持命令行参数。"""

from raya_maas.local_test import run_local_test

QUESTION = "用户说：帮我把明天下午3点的会议改到4点。这个请求的意图是什么？"
CANDIDATES = ["修改会议时间", "创建新会议", "取消会议", "查询会议详情"]


if __name__ == "__main__":
    run_local_test("text", QUESTION, CANDIDATES)
