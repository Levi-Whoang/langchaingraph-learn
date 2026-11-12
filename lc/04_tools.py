"""
04_tools.py —— 工具（Tool）：让模型能"做事"，不只是"说话"

【核心认知，先建立这个】
    模型的 tool calling 从来不执行任何东西。
    它只是输出一段 JSON：{"name": "get_weather", "args": {"city": "北京"}}
    真正去跑函数的是 agent 框架。所以"给模型加能力"= 把 Python 函数**描述清楚**给它看。

【所以写工具的重点不是逻辑，是描述】
    docstring 会被当成 description 一并发给模型。
    写"查询天气"和写"根据城市名查询该城市今日实时天气，返回温度与天气状况"，
    模型调对的概率差得很远。这是工具设计里最容易忽略、也最影响效果的地方。

运行：python lc/04_tools.py
"""

from typing import Literal

from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool, tool


# ---------------------------------------------------------------------------
# 1. 最简单的写法：@tool 装饰器
# ---------------------------------------------------------------------------
# 它干三件事：
#   · 函数名 → 工具名（模型看到的就是 get_weather）
#   · docstring 第一段 → description（模型决定"要不要用"的依据）
#   · 类型注解 → 参数 JSON Schema（模型据此填参数）
# 少任何一样，模型都会犯迷糊。
@tool
def get_weather(city: str) -> str:
    """根据城市名查询该城市的实时天气，返回天气状况和温度。"""
    fake_db = {"北京": "晴，18℃", "上海": "小雨，22℃", "深圳": "多云，27℃"}
    return fake_db.get(city, f"{city}：暂无数据")


# ---------------------------------------------------------------------------
# 2. 复杂参数：显式定义 args_schema
# ---------------------------------------------------------------------------
# 当参数有枚举、有范围、有默认值、有说明时，光靠类型注解不够。
# 用 Pydantic 定义 schema，字段的 description 会原样进入 JSON Schema，
# 模型能看到"这个参数到底是什么"，填错的概率显著降低。
class SearchInput(BaseModel):
    keyword: str = Field(description="搜索关键词，中文，不要带标点")
    limit: int = Field(default=3, description="返回条数，1-10 之间")
    sort: Literal["relevance", "date"] = Field(default="relevance", description="排序方式")


@tool(args_schema=SearchInput)
def search_notes(keyword: str, limit: int = 3, sort: str = "relevance") -> str:
    """在本地学习笔记中搜索包含关键词的片段。用户问"我之前记过什么"时使用。"""
    notes = [
        "第 1 天：LCEL 就是管道符，组件都是 Runnable。",
        "第 2 天：tool calling 模型不执行工具，只输出调用意图。",
        "第 3 天：agent 的循环 = 想 → 调工具 → 看结果 → 再想。",
        "第 4 天：checkpointer 才是多轮记忆的真正实现者。",
    ]
    hits = [n for n in notes if keyword in n][:limit]
    return "\n".join(hits) if hits else f"没有找到含「{keyword}」的笔记"


# ---------------------------------------------------------------------------
# 3. 从普通函数转成工具：StructuredTool.from_function
# ---------------------------------------------------------------------------
# 有时候函数已经被别处用了，不想加装饰器。用这个包一下就行。
def add(a: int, b: int) -> int:
    """两个整数相加。"""
    return a + b


add_tool = StructuredTool.from_function(
    func=add,
    name="add_numbers",
    description="计算两个整数的和，用于任何需要加法的地方。",
)


# ---------------------------------------------------------------------------
# 4. 单独调用工具（不经过模型）
# ---------------------------------------------------------------------------
# 调试工具时最有用的一步：先确认工具自己能跑对，再去接模型。
# 如果这里就错了，接上模型只会让你误以为是"模型不听话"。
print("=== 1. 直接调用 ===")
print("  ", get_weather.invoke({"city": "北京"}))
print("  ", add_tool.invoke({"a": 3, "b": 5}))

# ---------------------------------------------------------------------------
# 5. 工具对象里到底有什么（这些就是发给模型的东西）
# ---------------------------------------------------------------------------
print("\n=== 2. 工具的自描述 ===")
print("  name        =", search_notes.name)
print("  description =", search_notes.description)
print("  args(schema)=", search_notes.args)

# ---------------------------------------------------------------------------
# 6. 工具出错会怎样？
# ---------------------------------------------------------------------------
# 工具抛异常 → agent 会把异常信息包装成 ToolMessage 回给模型 → 模型看到报错，
# 常常会自己改参数重试。所以工具内部**不要**把异常吞掉返回"失败"，
# 让它抛出来，模型反而更聪明。
@tool
def divide(a: float, b: float) -> float:
    """两个数相除。"""
    if b == 0:
        raise ValueError("除数不能为 0")
    return a / b


print("\n=== 3. 工具的报错信息 ===")
try:
    divide.invoke({"a": 1, "b": 0})
except Exception as exc:
    print(f"  抛出 {type(exc).__name__}: {exc}")
    print("  → 这个错误会被 agent 转成 ToolMessage 交给模型，模型通常会改参数重试")

# ---------------------------------------------------------------------------
# 7. 绑定到模型（agent 内部干的就是这件事）
# ---------------------------------------------------------------------------
# bind_tools 之后，模型就"知道"有这些工具可用了，会在需要时输出 tool_calls。
tools = [get_weather, search_notes, add_tool, divide]

if __name__ == "__main__":
    from config import get_model

    llm = get_model()
    bound = llm.bind_tools(tools)
    print("\n=== 4. bind_tools 后模型的反应 ===")
    reply = bound.invoke("北京今天天气怎么样？")
    print("  tool_calls =", reply.tool_calls)
    print("  → 注意它只\"申请\"调用，没真执行。执行是 agent 的事（看 05_agent.py）")
