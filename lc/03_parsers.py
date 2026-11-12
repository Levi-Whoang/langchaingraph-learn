"""
03_parsers.py —— 输出解析器：把"一段文字"变成"程序能用的数据"

【为什么需要它】
    模型永远只会吐字符串。你说"给我 JSON"，它可能给你：
        ```json\n{...}\n```       ← 带 markdown 围栏
        好的，这是结果：{...}      ← 前面还唠一句
    手写 json.loads 迟早崩。OutputParser 就是干这个脏活的。
    LangChain 1.x 里更推荐 with_structured_output()，因为它不靠"哄"，
    而是靠 tool calling 强制模型按 schema 输出 —— 成功率完全不是一个量级。

运行：python lc/03_parsers.py
"""

from typing import Literal

from config import get_model

from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

llm = get_model(temperature=0.0)


# ---------------------------------------------------------------------------
# 1. StrOutputParser —— 最常用，把 AIMessage 抽出纯文本
# ---------------------------------------------------------------------------
# 不加它，你拿到的是 AIMessage 对象；加了，拿到的是 str。
# 90% 的链末尾都有它。
print("=== 1. StrOutputParser ===")
chain = ChatPromptTemplate.from_template("一句话解释：{topic}") | llm | StrOutputParser()
print("  ", chain.invoke({"topic": "依赖注入"}))


# ---------------------------------------------------------------------------
# 2. JsonOutputParser —— 拿 dict
# ---------------------------------------------------------------------------
# 它有两个作用：
#   a) 生成"格式说明书"（get_format_instructions），你可以塞进 prompt
#   b) 解析时自动剥掉 markdown 围栏和废话
parser = JsonOutputParser()

# 单独用它也行 —— parser 就是个普通 Runnable，invoke 一个字符串就出结果
raw = '```json\n{"name": "小w", "skills": ["LangChain", "LangGraph"]}\n```'
print("\n=== 2. JsonOutputParser ===")
print("  解析结果：", parser.invoke(raw))

# 链式用法：靠 partial 把格式要求注入 prompt 的 format_instructions 占位符
json_prompt = ChatPromptTemplate.from_messages([
    ("system", "你只输出 JSON，不要任何解释。\n{format_instructions}"),
    ("human", "{question}"),
]).partial(format_instructions=parser.get_format_instructions())


# ---------------------------------------------------------------------------
# 3. PydanticOutputParser —— 带校验的强类型输出
# ---------------------------------------------------------------------------
# JsonOutputParser 只保证"是合法 JSON"，字段缺了、类型错了它不管。
# Pydantic 版本会真的做校验，缺字段直接抛异常 —— 比在下游崩掉好得多。
class Movie(BaseModel):
    title: str = Field(description="电影名")
    year: int = Field(description="上映年份")
    genre: Literal["科幻", "剧情", "喜剧", "动作"] = Field(description="类型")


pyd_parser = PydanticOutputParser(pydantic_object=Movie)
print("\n=== 3. PydanticOutputParser ===")
print("  它会自动生成这段说明书（可以塞给模型看）：")
print("  " + pyd_parser.get_format_instructions().replace("\n", "\n  ")[:300] + " ...")

# 解析成功会得到 Movie 实例，不是 dict —— 有类型提示，IDE 能跳转
movie = pyd_parser.invoke('{"title": "星际穿越", "year": 2014, "genre": "科幻"}')
print("  解析结果：", movie, "| 类型：", type(movie).__name__)

# 解析失败会抛什么？值得看一眼，写 catch 的时候心里有数
try:
    pyd_parser.invoke('{"title": "星际穿越"}')
except Exception as exc:
    print(f"  字段缺失时抛：{type(exc).__name__}")


# ---------------------------------------------------------------------------
# 4. with_structured_output —— 1.x 的首选方案
# ---------------------------------------------------------------------------
# 区别在哪：
#     parser 方案 = 在 prompt 里写"请输出 JSON"，然后祈祷它听话
#     本方案      = 把 Pydantic 类变成一个 tool schema，逼模型"调工具填参数"
# 模型在 tool calling 这一层是被训练过的，格式正确率极高。
class Person(BaseModel):
    """从文本里抽取的人物信息。"""
    name: str = Field(description="姓名")
    age: int = Field(description="年龄，未知填 -1")
    city: str = Field(description="所在城市")


extractor = llm.with_structured_output(Person)
print("\n=== 4. with_structured_output ===")
print("  返回的是 Pydantic 实例：", extractor.invoke("我叫张三，28岁，住在杭州。"))
print("  直接点属性：", extractor.invoke("我叫张三，28岁，住在杭州。").city)

# 【注意】离线模式下这里拿到的是占位值，因为假模型不会真的理解文本。
# 配上真实 API Key 后再跑，你会看到 name/age/city 被正确抽出来。
