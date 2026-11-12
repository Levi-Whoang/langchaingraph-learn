"""
02_prompt_lcel.py —— 提示词模板 + LCEL 管道（LangChain 的"积木"）

【一句话理解】
    LCEL = LangChain Expression Language，就是那个竖线 `|`。
    它把"组件"用管道串起来：prompt | model | parser。
    每个组件都是 Runnable，都有 invoke / batch / stream / ainvoke 四个方法。
    串起来之后，整条链**自动**支持批量、流式、异步 —— 这是它比手写函数值钱的地方。

运行：python lc/02_prompt_lcel.py
"""

from config import get_model

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

llm = get_model(temperature=0.0)

# ---------------------------------------------------------------------------
# 1. 最基本的模板：把变量留成 {占位符}
# ---------------------------------------------------------------------------
# 为什么要模板？因为你不能拼字符串拼一辈子 —— 一旦变量来自用户输入，
# 拼接就等于把注入风险写进代码。模板会把变量放进正确的消息位置。
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位{role}，回答不超过{limit}个字。"),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()      # ← 这就是 LCEL，三个组件，两个竖线

print("=== 1. 最基础的链 ===")
print(chain.invoke({"role": "Python 讲师", "limit": "30", "question": "什么是闭包？"}))

# ---------------------------------------------------------------------------
# 2. 消息列表占位：接多轮历史
# ---------------------------------------------------------------------------
# 多轮对话里，"历史"本身是一个消息列表，不能塞进 {占位符}——
# 占位符只能填字符串。MessagesPlaceholder 就是专门给"消息列表"留的坑位。
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个话很少的助手。"),
    MessagesPlaceholder("history"),           # ← 这里会摊开成一个消息列表
    ("human", "{question}"),
])

print("\n=== 2. MessagesPlaceholder ===")
print(chat_prompt.invoke({
    "history": [("human", "我在学 LangChain"), ("ai", "好")],
    "question": "我刚才说我在学什么？",
}).to_messages()[-1])

# ---------------------------------------------------------------------------
# 3. partial：把一部分变量先"焊死"
# ---------------------------------------------------------------------------
# 场景：system 里那句话永远不变，每次调用都传一遍很烦。
# partial 返回一条新链，剩下的变量才需要你传。
locked = (prompt.partial(role="严格的代码审查员", limit="20") | llm | StrOutputParser())
print("\n=== 3. partial 预绑定变量 ===")
print(locked.invoke({"question": "Python 的 list 和 tuple 区别？"}))

# ---------------------------------------------------------------------------
# 4. RunnableParallel：一次输入，多路并发
# ---------------------------------------------------------------------------
# 写 dict 也能达到同样效果，但用 RunnableParallel 意图更明确，而且真的并发执行。
# 用途：同一段文本同时做"总结 / 翻译 / 关键词提取"。
summary_chain = (
    RunnableParallel(
        summary=(ChatPromptTemplate.from_template("用一句话总结：{text}") | llm | StrOutputParser()),
        keywords=(ChatPromptTemplate.from_template("提取3个关键词，逗号分隔：{text}") | llm | StrOutputParser()),
    )
)
print("\n=== 4. RunnableParallel 并行分支 ===")
print(summary_chain.invoke({"text": "LangChain 是一个用于构建大模型应用的框架，支持链式组合、工具调用和记忆。"}))

# ---------------------------------------------------------------------------
# 5. RunnableLambda：把普通函数塞进管道
# ---------------------------------------------------------------------------
# 现实里总有一步不是模型干的（清洗数据、查库、格式化）。
# 用 RunnableLambda 一包，它立刻获得 invoke/batch/stream 全套能力。
clean = RunnableLambda(lambda x: x.strip().replace("。", ""))

post_chain = (
    {"text": RunnablePassthrough()}       # 把输入原样透传并包成 dict
    | ChatPromptTemplate.from_template("用5个字形容：{text}")
    | llm
    | StrOutputParser()
    | clean                                # ← 普通函数也能接在管道尾部
)

print("\n=== 5. RunnableLambda ===")
print(repr(post_chain.invoke("  今天天气特别好。  ")))

# ---------------------------------------------------------------------------
# 6. .batch()：一次跑多条 —— 这是 LCEL 白送的能力
# ---------------------------------------------------------------------------
# 你自己写 for 循环调模型，是一条一条串行等；batch 帮你并发发出去。
print("\n=== 6. batch 批量 ===")
for ans in chain.batch([{"role": "极简主义回答者", "limit": "8", "question": q}
                        for q in ["什么是 HTTP？", "什么是 TCP？"]]):
    print("  ", ans)
