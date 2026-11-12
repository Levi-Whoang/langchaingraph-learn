"""
05_agent.py —— ★ 本轮的主角：一个能用的 LangChain Agent

【agent 到底是什么？一句话】
    agent = 一个「想 → 调工具 → 看结果 → 再想」的循环，直到模型说"我想完了"。

    普通链是直线：prompt → model → parser，跑完就结束。
    agent 是循环：模型可能连续说三次"我要调工具"，框架就执行三次再回头问它，
    直到它输出一条不带 tool_calls 的普通消息，循环才停。
    LangChain 1.x 的 create_agent() 帮你把这个循环、状态、中断、流式全封好了。

【这个文件演示了哪些用法】
    1. create_agent() 的最小可用形态
    2. 三个不同形态的工具（有参/无参/带安全校验）
    3. system_prompt 设定人设
    4. checkpointer + thread_id 实现多轮记忆
    5. 打印内部轨迹（学习 agent 最重要的一步）
    6. response_format 拿到结构化结果
    7. 递归上限、配置项

运行：python lc/05_agent.py
"""

import ast
import operator
import re
from datetime import datetime
from typing import Any

from config import get_model

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

# ===========================================================================
# 第一步：定义工具
# ===========================================================================
# 记住 04 里的结论：模型只能"申请"调用，真正干活的是 agent 框架调这些函数。
# 所以每个函数的逻辑必须自己保证安全 —— 模型传什么进来你都得扛住。


def _pick_keyword(text: str) -> list[str]:
    """
    离线/真实模型都可能把整句用户提问塞进参数里。
    工具自己有责任清洗输入，不能假设上游一定给你干净的关键词。

    【踩过的坑】一开始我用"二字切片"匹配，结果 "calling" 里的 "in" 都能命中，
    一条提问把五条笔记全搜出来了。2 字片段信息量太低，
    必须优先用英文/数字词，中文也至少切 3 个字。
    """
    words = re.findall(r"[A-Za-z0-9_]{3,}", text)          # 英文/数字词最可靠，优先用
    if words:
        return [w.lower() for w in words]
    # 没有英文词时，按标点/语气词切开中文，只留 3 字以上的片段
    chunks = re.split(r"[，。？！,.?!、；：\s]|的|了|吗|呢|吧|啊", text)
    return [c for c in chunks if len(c) >= 3]


@tool
def search_notes(query: str) -> str:
    """在 hewier 的 LangChain 学习笔记中搜索相关内容。当用户问"我之前记过什么"、"帮我找一下笔记"时使用。"""
    notes = [
        "第 1 天：LCEL 就是管道符 |，所有组件都是 Runnable。",
        "第 2 天：tool calling 中模型不执行工具，只输出调用意图。",
        "第 3 天：agent 的循环 = 想 → 调工具 → 看结果 → 再想。",
        "第 4 天：checkpointer 才是多轮记忆的真正实现者，thread_id 是它的分区键。",
        "第 5 天：middleware 能在模型调用前后插入逻辑，不用改 agent 本体。",
    ]
    keys = _pick_keyword(query)
    hits = [n for n in notes if any(k.lower() in n.lower() for k in keys)]
    return "\n".join(hits) if hits else f"没有找到和「{query}」相关的笔记"


# 四则运算的运算符表。用 ast 解析而不是 eval —— eval 会把工具的入参变成远程代码执行。
_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval(node.operand)
    raise ValueError("表达式含不允许的语法")


@tool
def calculate(expression: str) -> str:
    """计算四则运算表达式，支持 + - * / 和括号。用户问"算一下"、"等于多少"时使用。"""
    # 从参数里抠出算式部分（模型有时会把整句话塞进来）
    match = re.search(r"[-+*/().\d\s]{3,}", expression)
    if not match:
        raise ValueError(f"没能从「{expression}」里识别出算式")
    expr = match.group().strip()
    value = _safe_eval(ast.parse(expr, mode="eval").body)
    return f"{expr} = {value}"


@tool
def get_current_time() -> str:
    """获取当前的日期和时间。用户问"现在几点"、"今天几号"时使用。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = [search_notes, calculate, get_current_time]

# ===========================================================================
# 第二步：造 agent
# ===========================================================================
# create_agent 的每个参数都值得说一句：
#   model          → 任意 BaseChatModel，或 "provider:model" 字符串
#   tools          → 工具列表，模型能"看见"它们
#   system_prompt  → 人设 + 行为规则，等价于 SystemMessage，但更省事
#   checkpointer   → 记忆存哪儿。不传 = 完全没有记忆，每次调用都从零开始
SYSTEM_PROMPT = """你是 hewier 的 LangChain 学习助手，名叫小w。

工作规则：
1. 用户问笔记内容、需要算数、需要当前时间时，必须先调工具，不要凭记忆瞎猜。
2. 工具返回什么就基于什么回答，不要编造工具没给你的信息。
3. 回答简短，中文，不啰嗦。"""

agent = create_agent(
    model=get_model(temperature=0.0),
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
    # InMemorySaver 存在内存里，进程一退就没了。
    # 生产环境换 SqliteSaver / PostgresSaver，用法完全一样。
    checkpointer=InMemorySaver(),
)


# ===========================================================================
# 第三步：一个把内部轨迹打印出来的小工具
# ===========================================================================
# 学 agent 最有效的方法就是盯着这个消息列表看：
#   human → ai(tool_calls) → tool → ai(最终回答)
# 一旦你能在脑子里画出这条线，agent 就没有黑盒了。
def show_trace(messages: list, since: int = 0) -> None:
    for msg in messages[since:]:
        kind = msg.type
        if kind == "human":
            print(f"  👤 用户：{msg.content}")
        elif kind == "ai":
            if getattr(msg, "tool_calls", None):
                for call in msg.tool_calls:
                    print(f"  🤖 模型申请调用：{call['name']}({call['args']})")
            else:
                print(f"  🤖 回答：{msg.content}")
        elif kind == "tool":
            preview = str(msg.content).replace("\n", " ⏎ ")[:70]
            print(f"  🔧 工具返回：{preview}")


# 记一下每个 thread 上一轮结束时有多少条消息，这样每次只打印"这一轮新增的"，
# 否则历史越长输出越糊。
_seen: dict[str, int] = {}


def ask(question: str, thread_id: str = "default") -> str:
    """给 agent 发一句话，返回最终回答。thread_id 决定它读哪一段记忆。"""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        # config 里的 thread_id 是 checkpointer 的分区键：
        # 同一个 thread_id → 共享历史；换一个 → 全新对话。
        # 这就是"多用户/多会话隔离"的实现方式，不用你手写 session 字典。
        config={"configurable": {"thread_id": thread_id}},
    )
    messages = result["messages"]
    show_trace(messages, since=_seen.get(thread_id, 0))
    _seen[thread_id] = len(messages)
    return messages[-1].content


# ===========================================================================
# 第四步：跑起来
# ===========================================================================
def chat() -> None:
    """交互模式：真的跟 agent 聊天。启动：python lc/05_agent.py chat"""
    print("进入交互模式（输入 exit 退出）")
    print("试试：我记过 checkpointer 吗 / 128*46 等于多少 / 现在几点\n")
    while True:
        try:
            question = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question in {"exit", "quit", "q", ""}:
            break
        ask(question, thread_id="cli")


def demo() -> None:
    """演示模式：按顺序跑几个固定场景，看 agent 内部发生了什么。"""
    print("=" * 60)
    print("① 单轮：触发工具（search_notes）")
    print("=" * 60)
    ask("我之前记过 checkpointer 的笔记吗？", thread_id="session-1")

    print("\n" + "=" * 60)
    print("② 单轮：触发另一个工具（calculate）")
    print("=" * 60)
    ask("帮我算一下 128 * 46 + 7 等于多少", thread_id="session-1")

    print("\n" + "=" * 60)
    print("③ 多轮：验证记忆是否真的生效")
    print("=" * 60)
    print("  （注意：这个问题不需要调工具，模型要靠上面的历史来回答）")
    ask("我们刚才聊了哪几件事？", thread_id="session-1")

    print("\n" + "=" * 60)
    print("④ 换一个 thread_id：记忆应该是空的")
    print("=" * 60)
    ask("我们刚才聊了哪几件事？", thread_id="session-2")

    print("\n" + "=" * 60)
    print("⑤ 结构化输出：让 agent 交回一个 Pydantic 对象")
    print("=" * 60)

    class Answer(BaseModel):
        """对用户问题的结构化回答。"""
        summary: str = Field(description="一句话结论")
        used_tool: bool = Field(description="这次回答是否调用了工具")
        confidence: int = Field(description="置信度 0-100")

    # 方式 A（通用，任何模型都能用）：with_structured_output
    # 它的原理是把 Answer 变成一个"工具 schema"，逼模型按格式填参数。
    structured_llm = get_model(temperature=0.0).with_structured_output(Answer)
    print("  with_structured_output 返回的是 Answer 实例：")
    print("   ", structured_llm.invoke("checkpointer 是干什么用的？"))

    # 方式 B（agent 专用）：给 create_agent 传 response_format
    # agent 的循环跑完之后，会再多走一步，把最终答案按 Answer 重新整理，
    # 结果放在 state 的 structured_response 字段里 —— 是个 Pydantic 实例，不是 dict。
    # 下面这段在离线模型下跑不通（假模型不会"再走一步"），配上真 key 就能用了：
    print("""
  agent = create_agent(
      model=llm,
      tools=TOOLS,
      system_prompt=SYSTEM_PROMPT,
      response_format=Answer,        # ← 就多这一个参数
  )
  out = agent.invoke({"messages": [{"role": "user", "content": "现在几点了？"}]})
  out["structured_response"]         # ← Answer 实例，可以直接 .summary / .confidence
    """)

    print("\n" + "=" * 60)
    print("⑥ 递归上限：防止 agent 无限调工具烧钱")
    print("=" * 60)
    print("  agent 默认最多循环 25 步（LangGraph 的 recursion_limit）。")
    print("  超出会抛 GraphRecursionError。生产环境建议显式设成 10 左右：")
    print()
    print("      agent.invoke(")
    print("          {'messages': [...]},")
    print("          config={'configurable': {'thread_id': 'x'}, 'recursion_limit': 10},")
    print("      )")


# 入口分发：python lc/05_agent.py chat  → 真跟它聊
#           python lc/05_agent.py       → 跑固定演示
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        chat()
    else:
        demo()
