"""
01_messages.py —— 消息（Message）：LangChain 里"对话"的最基本单位

【一句话理解】
    在 LangChain 里，你跟模型的一次对话不是一个字符串，而是一个**消息列表**。
    模型没有记忆，所谓"多轮对话"就是每次把完整的历史列表再发一遍。

【四种消息，四种角色】
    SystemMessage  —— 设定人设/规则，永远放第一条。用户看不见，但它支配语气和行为
    HumanMessage   —— 用户说的话
    AIMessage      —— 模型说的话（可能带 tool_calls，即"我要调工具"）
    ToolMessage    —— 工具执行完的结果（必须带 tool_call_id，告诉模型"这是哪次调用的回话"）

运行：python lc/01_messages.py
"""

from config import get_model

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# ---------------------------------------------------------------------------
# 1. 手搓一个消息列表 —— 这是最原始也最本质的用法
# ---------------------------------------------------------------------------
# 为什么要有 SystemMessage？因为模型的"人设"不是靠求它，是靠这个高优先级角色。
# 生产环境里，护栏（不许聊政治、必须输出 JSON）基本都写在这里。
history = [
    SystemMessage(content="你是一个只用一句话回答的助手。"),
    HumanMessage(content="我叫 hewier，在学 LangChain。"),
    AIMessage(content="记住了。"),
    HumanMessage(content="我叫什么？"),
]

print("=== 1. 消息列表长什么样 ===")
for msg in history:
    # msg.type 是身份标签："system" / "human" / "ai" / "tool"
    print(f"  {msg.type:6} | {msg.content}")

# ---------------------------------------------------------------------------
# 2. 把消息列表丢给模型
# ---------------------------------------------------------------------------
# 注意：模型接口叫 invoke，不叫 chat / generate。
# 这是 LangChain 的统一约定 —— 所有"能被调用一次的东西"都叫 Runnable，都有 invoke。
llm = get_model(temperature=0.0)
reply = llm.invoke(history)
print("\n=== 2. 模型回复 ===")
print("  ", reply.content)

# ---------------------------------------------------------------------------
# 3. 简写：dict 形式
# ---------------------------------------------------------------------------
# 每次手写 HumanMessage(...) 太啰嗦，LangChain 接受这种简写，
# 内部会自动转成对应消息类型。agent 里传参几乎都用这个写法。
short = [
    {"role": "system", "content": "你是 LangChain 助教。"},
    {"role": "user", "content": "用一句话说 LangChain 是什么。"},
]
print("\n=== 3. dict 简写 ===")
print("  ", llm.invoke(short).content)

# ---------------------------------------------------------------------------
# 4. AIMessage 里的 tool_calls —— 理解 agent 的关键
# ---------------------------------------------------------------------------
# 模型想调工具时，不是"执行"工具，而是**输出一段结构化的意图**：
#     "我想调 get_weather，参数是 {'city': '北京'}，编号是 call_abc"
# 真正执行工具的是 agent 框架（或你自己写的循环）。
# 这就是为什么工具调用永远不会越权 —— 模型只能"申请"，不能"动手"。
tool_request = AIMessage(
    content="",
    tool_calls=[{"name": "get_weather", "args": {"city": "北京"}, "id": "call_abc", "type": "tool_call"}],
)
print("\n=== 4. 模型的工具调用意图 ===")
print("  ", tool_request.tool_calls)

# 工具执行完后，必须用同一条 id 回话，否则模型对不上号
tool_result = ToolMessage(content="北京：晴，18℃", tool_call_id="call_abc")
print("  ", tool_result.type, "->", tool_result.content)

# ---------------------------------------------------------------------------
# 5. 消息对象还能干什么
# ---------------------------------------------------------------------------
print("\n=== 5. 常用属性 ===")
print("  reply.type          =", reply.type)              # ai
print("  reply.id            =", reply.id)                # 每条消息有唯一 id
print("  reply.response_metadata =", reply.response_metadata)  # token 用量等原始元数据
print("  history[-1].text    =", history[-1].text)        # .text 只取纯文本（多模态时有用）
