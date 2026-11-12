"""
07_middleware.py —— 中间件：不改 agent 本体，往循环里插逻辑

【为什么需要它（1.x 最重要的新增能力）】
    你想给 agent 加"记日志、限流、脱敏、超长自动摘要、人工审批"……
    以前只能改 agent 内部代码，或者在外面套一层，越套越乱。
    middleware 就是官方给的"插槽"，在固定时机被调用：

        before_agent    → 整个 agent 开始前（只跑一次）
        before_model    → 每次调模型前
        after_model     → 每次调模型后
        wrap_tool_call  → 把工具调用包起来（前后都能插手，还能改写/拦截）
        after_agent     → 整个 agent 结束后

    它跟"装饰器"的区别：中间件拿得到完整的 agent 状态（消息列表、工具调用），
    所以能做"看到条件才触发"的事，而不只是无脑前后打印。

运行：python lc/07_middleware.py
"""

import time

from config import get_model

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ToolCallLimitMiddleware,
    after_model,
    before_model,
)
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# 1. 类写法：一次定义多个钩子
# ---------------------------------------------------------------------------
# 适合"一整套审计逻辑"绑在一起。注意钩子返回 None = 不改状态；
# 返回 dict = 往 state 里打补丁（比如 {"messages": [...]} 追加消息）。
class AuditMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        print(f"  [audit] 准备调模型，当前历史 {len(state['messages'])} 条")
        return None

    def after_model(self, state, runtime):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            names = [c["name"] for c in last.tool_calls]
            print(f"  [audit] 模型决定调工具：{names}")
        else:
            print(f"  [audit] 模型给出最终回答（{len(str(last.content))} 字）")
        return None


# ---------------------------------------------------------------------------
# 2. 装饰器写法：只加一个小钩子，不值得写类
# ---------------------------------------------------------------------------
# before_model 装饰的函数签名叫 (state, runtime)，跟类方法一样。
@before_model
def inject_reminder(state, runtime):
    """每轮都在最前面塞一句提醒 —— 这是"动态 system prompt"的最简实现。"""
    # 只在第一轮注入，否则历史会越堆越长
    if any(m.type == "system" for m in state["messages"]):
        return None
    print("  [reminder] 注入了一条临时提醒")
    # 返回的是 state 的增量更新
    return {"messages": [{"role": "system", "content": "（提醒）回答请控制在 50 字以内。"}]}


@after_model
def count_tokens(state, runtime):
    """粗略统计一下每轮输出长度，真实项目里换成 tiktoken 精确计数。"""
    last = state["messages"][-1]
    print(f"  [cost] 本轮输出约 {len(str(last.content))} 字符")
    return None


# ---------------------------------------------------------------------------
# 3. 一个真正有用的中间件：工具调用计时 + 异常兜底
# ---------------------------------------------------------------------------
# wrap_tool_call 能拿到 request（含工具名和参数）和 handler（真正执行函数）。
# 必须自己调 handler(request)，不调 = 工具不会执行。
class ToolLogger(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        name = request.tool_call["name"]
        print(f"  [tool] → {name}({request.tool_call['args']})")
        start = time.time()
        try:
            result = handler(request)
        except Exception as exc:
            # 这里可以把异常写进监控，再决定要不要吞掉
            print(f"  [tool] ✗ {name} 失败：{type(exc).__name__}: {exc}")
            raise
        print(f"  [tool] ← {name} 耗时 {time.time() - start:.3f}s")
        return result


# ---------------------------------------------------------------------------
# 4. 准备工具和 agent
# ---------------------------------------------------------------------------
@tool
def search_notes(query: str) -> str:
    """在 LangChain 学习笔记中搜索内容。用户问"我之前记过什么"时使用。"""
    notes = [
        "第 1 天：LCEL 就是管道符。",
        "第 5 天：middleware 能在模型调用前后插入逻辑，不用改 agent 本体。",
    ]
    hits = [n for n in notes if any(w.lower() in n.lower() for w in query.split())]
    return "\n".join(hits) if hits else f"没找到「{query}」"


@tool
def get_current_time() -> str:
    """获取当前的日期和时间。用户问「现在几点」时使用。"""
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")


if __name__ == "__main__":
    print("=" * 60)
    print("① 自定义中间件：审计 + 提醒 + 工具日志")
    print("=" * 60)

    agent = create_agent(
        model=get_model(temperature=0.0),
        tools=[search_notes, get_current_time],
        system_prompt="你是学习助手，回答简短。",
        # 顺序有意义：列表里越靠前，包装层越靠外（先进后出）
        middleware=[AuditMiddleware(), inject_reminder, count_tokens, ToolLogger()],
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "现在几点了？"}]})
    print("  最终回答：", result["messages"][-1].content)

    print("\n" + "=" * 60)
    print("② 内置中间件：ToolCallLimitMiddleware 限制工具调用次数")
    print("=" * 60)
    print("  场景：模型陷入「调工具-不满意-再调」的死循环，你会被账单教做人。")
    print("  run_limit = 单次 invoke 内最多调几次；thread_limit = 整个会话上限。")

    limited_agent = create_agent(
        model=get_model(temperature=0.0),
        tools=[search_notes, get_current_time],
        system_prompt="你是学习助手。",
        middleware=[
            ToolCallLimitMiddleware(run_limit=2, exit_behavior="end"),
            ToolLogger(),
        ],
    )
    out = limited_agent.invoke({"messages": [{"role": "user", "content": "现在几点了？顺便查下笔记"}]})
    print("  最终回答：", out["messages"][-1].content)

    print("\n" + "=" * 60)
    print("③ 其它内置中间件（按需取用，不用自己写）")
    print("=" * 60)
    print("""
  SummarizationMiddleware     历史太长自动摘要，避免撑爆上下文窗口
                              SummarizationMiddleware(model=llm, trigger=("messages", 20))
  ModelCallLimitMiddleware    限制模型调用次数（防死循环烧钱）
  ToolRetryMiddleware         工具失败自动重试，可配退避策略
  PIIMiddleware               自动脱敏手机号/邮箱/身份证
  HumanInTheLoopMiddleware    工具执行前暂停，等人工审批（危险操作必备）
  TodoListMiddleware          给 agent 一个待办清单，长任务拆解用
  ModelFallbackMiddleware     主模型挂了自动切备用模型
  LLMToolSelectorMiddleware   工具太多（50+）时，先让模型筛出相关的那几个

  这些的用法都一样：塞进 middleware=[...] 列表里就行。
""")
