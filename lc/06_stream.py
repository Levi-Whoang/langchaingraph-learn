"""
06_stream.py —— 流式输出：让用户不用干等

【为什么必须流式】
    invoke() 是"憋完整段再给你"，一个长回答可能等 10 秒。
    stream() 是"吐一个字给你一个字"，首字延迟能压到几百毫秒。
    用户体验的差别不是一点点 —— 这也是 ChatGPT 看起来"快"的真正原因。

【三个层次，别搞混】
    llm.stream()       → 模型层的 token 流
    chain.stream()     → 链的 token 流（前提：链末端不要求完整输入）
    agent.stream()     → agent 的**步骤**流，比 token 更粗，但信息量大得多

运行：python lc/06_stream.py
"""

import time

from config import get_model

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = get_model(temperature=0.0)

# ---------------------------------------------------------------------------
# 1. 模型层：llm.stream()
# ---------------------------------------------------------------------------
# 拿到的是 AIMessageChunk，不是字符串。想拼成完整回答要自己加。
# 注意 print 的 end="" + flush=True —— 不 flush 的话缓冲区会攒着，看着还是"一次性出来"。
print("=== 1. llm.stream() ===")
t0 = time.time()
first_token_at = None
for chunk in llm.stream("用三句话介绍 LangChain。"):
    if first_token_at is None:
        first_token_at = time.time() - t0     # 首字延迟（TTFT），比总耗时更影响体感
    print(chunk.content, end="", flush=True)
print(f"\n  [首字 {first_token_at:.3f}s / 总计 {time.time() - t0:.3f}s]")

# 把 chunk 拼起来就是完整消息：直接用 + 号（AIMessageChunk 重载了加法）
print("\n=== 1b. 拼回完整消息 ===")
full = None
for chunk in llm.stream("说一句很短的话"):
    full = chunk if full is None else full + chunk
print("  拼起来的内容：", repr(full.content), "| 类型：", type(full).__name__)

# ---------------------------------------------------------------------------
# 2. 链层：chain.stream()
# ---------------------------------------------------------------------------
# 只有链上每个组件都能"增量处理"时才真的流式。
# StrOutputParser 能流；JsonOutputParser 不能（它要看到完整 JSON 才能解析），
# 所以带 JsonOutputParser 的链会自动退化成一次性输出 —— 这是常见坑。
print("\n=== 2. chain.stream() ===")
chain = ChatPromptTemplate.from_template("用两个字回答：{q}") | llm | StrOutputParser()
for piece in chain.stream({"q": "你是谁"}):
    print(piece, end="", flush=True)
print()

# ---------------------------------------------------------------------------
# 3. agent 层：stream_mode 决定"流什么"
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    @tool
    def get_time() -> str:
        """获取当前的日期和时间。用户问「现在几点」「今天几号」时使用。"""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")

    agent = create_agent(model=llm, tools=[get_time], system_prompt="你是助手，回答简短。")
    question = {"messages": [{"role": "user", "content": "现在几点了？"}]}

    print("\n=== 3a. stream_mode='values'：每步吐出完整 state ===")
    # 好处：能完整看到消息列表怎么一点点长出来。
    # 坏处：状态越大，重复传输越多。
    for i, state in enumerate(agent.stream(question, stream_mode="values")):
        last = state["messages"][-1]
        detail = last.tool_calls if getattr(last, "tool_calls", None) else str(last.content)[:40]
        print(f"  第{i}帧 [{last.type}] {detail}")

    print("\n=== 3b. stream_mode='updates'：只吐增量（哪个节点产出了什么）===")
    # 生产环境最常用：只关心"有新东西了"，不在乎全量状态。
    for update in agent.stream(question, stream_mode="updates"):
        for node_name, payload in update.items():
            print(f"  节点 {node_name} → {list(payload.keys())}")

    print("\n=== 3c. stream_mode='messages'：token 级 + 元数据 ===")
    # 元组形式 (chunk, metadata)，metadata 告诉你这个 token 来自哪个节点、
    # 哪次模型调用。做前端时就是靠它把"思考中"和"回答中"区分开。
    for chunk, metadata in agent.stream(question, stream_mode="messages"):
        if chunk.content:
            print(f"[{metadata.get('langgraph_node', '?')}] {chunk.content}", end=" ")

    print("\n\n=== 3d. 多模式同时流 ===")
    # stream_mode 可以传列表，拿到的是元组 (mode, data)
    for mode, data in agent.stream(question, stream_mode=["updates", "values"]):
        print(f"  {mode}: {type(data).__name__}")

    print("\n=== 3e. 异步版 astream ===")
    print("  （FastAPI / WebSocket 场景用这个，别在异步框架里调同步 stream，会阻塞事件循环）")

    import asyncio

    async def demo_async():
        async for chunk in agent.astream(question, stream_mode="messages"):
            piece, _meta = chunk
            if piece.content:
                print(piece.content, end="", flush=True)
        print()

    asyncio.run(demo_async())
