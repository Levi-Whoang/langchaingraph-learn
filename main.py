"""
main.py —— 项目总入口

用法：
    python main.py            # 列出所有章节
    python main.py 5          # 跑第 5 章（agent）

每个章节都是独立可跑的小脚本，改一个不会影响别的。
想直接跑某个文件也行：python lc/05_agent.py
"""

import subprocess
import sys
from pathlib import Path

LC_DIR = Path(__file__).parent / "lc"

CHAPTERS = {
    "1": ("01_messages.py", "消息类型：System / Human / AI / Tool，多轮对话的本质"),
    "2": ("02_prompt_lcel.py", "提示词模板 + LCEL 管道 + 并行分支 + 批量"),
    "3": ("03_parsers.py", "输出解析：Str / Json / Pydantic / 结构化输出"),
    "4": ("04_tools.py", "工具：@tool 三种写法 + schema 设计 + 报错行为"),
    "5": ("05_agent.py", "★ Agent：工具循环 + 记忆 + 轨迹 + 结构化输出"),
    "6": ("06_stream.py", "流式输出：token 流 / 步骤流 / 异步"),
    "7": ("07_middleware.py", "中间件：往 agent 循环里插逻辑"),
}


def main() -> None:
    if len(sys.argv) < 2:
        print("LangChain 学习目录（先看第 5 章）\n")
        for key, (_, desc) in CHAPTERS.items():
            print(f"  {key}. {desc}")
        print("\n用法：python main.py 5")
        return

    key = sys.argv[1]
    if key not in CHAPTERS:
        print(f"没有第 {key} 章，可选：{', '.join(CHAPTERS)}")
        raise SystemExit(1)

    script = LC_DIR / CHAPTERS[key][0]
    print(f"▶ 运行 {script.name}\n")
    # 用子进程跑，保证每个章节的环境和单独执行时完全一致
    subprocess.run([sys.executable, str(script)], check=False)


if __name__ == "__main__":
    main()
