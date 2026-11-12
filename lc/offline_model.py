"""
offline_model.py —— 离线假模型：没有 API Key 也能把 LangChain 整条链路跑通。

【为什么要这个东西】
    学 LangChain 最痛苦的不是 API 复杂，是"没 key 就跑不了，跑不了就学不动"。
    这个类实现的是 BaseChatModel 接口 —— 也就是 LangChain 眼里"合格的模型"。
    它不联网，靠几条土规则假装成模型，但它在**协议层是完全真实的**：
      · 会返回 AIMessage
      · 会返回 tool_calls（工具调用）
      · 支持 bind_tools / 流式输出
    所以 agent 的循环、中间件、流式打印……全都会真实地跑一遍。
    等你配好 DEEPSEEK_API_KEY，把 config.get_model() 一换，业务代码不用动。

【这里能学到的 LangChain 知识】
    · 自定义模型 = 继承 BaseChatModel + 实现 _generate / _llm_type
    · 流式输出 = 覆盖 _stream，yield 一个一个的 ChatGenerationChunk
    · bind_tools 约定 = 返回一个"记住了工具列表"的新实例（不能改自己，Runnable 要求无副作用）
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Iterator, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field


def _overlap_score(source: str, target: str) -> int:
    """
    算「source 和 target 有多少共同片段」，用来猜用户想调哪个工具。
    真模型是理解语义的，这里只能退化成"字面重合度" —— 够用就行。
    """
    if not source or not target:
        return 0
    score = 0
    n = len(source)
    for size in (4, 3, 2):                      # 长片段更值钱，优先计分
        for i in range(n - size + 1):
            if source[i : i + size] in target:
                score += size
    return score


def _is_schema_class(obj: Any) -> bool:
    """判断这是不是一个 Pydantic 模型类（with_structured_output 会直接把类塞进来）。"""
    return isinstance(obj, type) and hasattr(obj, "model_json_schema")


def _structured_parts(obj: Any) -> tuple[str, dict[str, Any]]:
    """从一个 Pydantic 类 / 工具对象里取出 (名字, properties)。"""
    if _is_schema_class(obj):
        full = obj.model_json_schema()
        return obj.__name__, full.get("properties") or {}
    props, _ = _schema_parts(obj)
    return getattr(obj, "name", "extract"), props


def _schema_parts(tool_obj: Any) -> tuple[dict[str, Any], list[str]]:
    """
    取一个工具的 (参数说明, 必填参数名)。

    【踩过的坑】BaseTool.args 返回的是「properties 部分」，不是完整 JSON Schema，
    里面没有 "required" 字段。想要必填信息必须走 args_schema.model_json_schema()。
    这里把两种形态都兼容掉。
    """
    args_schema = getattr(tool_obj, "args_schema", None)
    if args_schema is not None and hasattr(args_schema, "model_json_schema"):
        full = args_schema.model_json_schema()
        props = full.get("properties") or {}
        required = full.get("required") or list(props)
        return props, required
    props = getattr(tool_obj, "args", {}) or {}
    return props, list(props)


class OfflineChatModel(BaseChatModel):
    """一个不联网的假模型。对外表现和 ChatOpenAI 一样。"""

    temperature: float = 0.0
    # bind_tools 之后，工具列表存在这里。注意是"新实例"带过来的，不是共享状态。
    bound_tools: list[Any] = Field(default_factory=list)
    # with_structured_output() 内部会偷偷 bind 一个叫 extract 的"抽取工具"。
    # 这里把它认出来，好让结构化输出这一节也能离线跑通。
    structured_tool: Any = None

    # -- 必须实现：给模型起个类型名，日志/追踪里会显示 ----------------------
    @property
    def _llm_type(self) -> str:
        return "offline-mock"

    # -- bind_tools：把工具"绑"到模型上 ------------------------------------
    # LangChain 的约定：不要改自己，返回一个副本。
    # 因为同一个 llm 对象可能被多个 chain 共用，就地修改会互相污染。
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "OfflineChatModel":
        tools = list(tools)
        # 找找有没有「结构化输出用的 schema」。
        # 注意 with_structured_output 传进来的是 Pydantic **类**，不是工具实例。
        structured = next((t for t in tools if _is_schema_class(t)), None)
        if structured is None:
            structured = next(
                (
                    t for t in tools
                    if "extract" in getattr(t, "name", "").lower()
                    or "Extract" in (getattr(t, "description", "") or "")
                ),
                None,
            )
        return self.model_copy(update={"bound_tools": tools, "structured_tool": structured})

    # -- 核心：给定消息历史，产出一条回复 ----------------------------------
    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self._decide(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    # -- 流式：把上面那条回复拆成一个个字吐出去 ----------------------------
    # 跑 06_stream.py 时会看到"打字机"效果。真模型是一个 token 一个 token 吐，
    # 这里简化成一个汉字一个 chunk，但对你写调用方代码来说完全一样。
    def _stream(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message = self._decide(messages)

        # 工具调用没法"逐字"流式，直接整块吐出去，并带上 tool_call_chunks 标记
        if message.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": call["name"],
                            "args": json.dumps(call["args"], ensure_ascii=False),
                            "id": call["id"],
                            "index": i,
                        }
                        for i, call in enumerate(message.tool_calls)
                    ],
                )
            )
            return

        text = message.content if isinstance(message.content, str) else str(message.content)
        for char in text:
            yield ChatGenerationChunk(message=AIMessageChunk(content=char))

    # =====================================================================
    # 下面是"土味大脑"：决定这次该说话还是该调工具
    # =====================================================================
    def _decide(self, messages: Sequence[BaseMessage]) -> AIMessage:
        last = messages[-1]

        # 中间件可能往末尾追加 system 消息（比如动态注入提醒），
        # 那玩意儿不是"用户说的话"，得往前找到真正的输入。
        if isinstance(last, BaseMessage) and last.type == "system":
            for m in reversed(messages):
                if m.type in ("human", "tool"):
                    last = m
                    break

        # 情况 0：被 with_structured_output 绑过，且只绑了这一个"抽取工具"
        #         → 直接按 schema 编一份符合格式的假数据交差。
        #         注意这个分支必须放在最前面，否则会被下面的工具匹配抢走。
        if self.structured_tool is not None and len(self.bound_tools) == 1 and not isinstance(last, ToolMessage):
            schema_name, props = _structured_parts(self.structured_tool)
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": schema_name,
                    "args": self._fake_args(props),
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                }],
            )

        # 情况 1：上一条是工具返回结果 → 说明该总结了，给出最终回答
        if isinstance(last, ToolMessage):
            # 【关键】只收「本轮」的工具结果。
            # 如果无脑收全部历史里的 ToolMessage，第二轮回答会把第一轮的
            # 工具结果一起复述出来 —— 真模型不会犯这错，但这里必须自己收拾。
            current_round: list[str] = []
            for m in reversed(messages):
                if isinstance(m, ToolMessage):
                    current_round.append(str(m.content))
                else:
                    break
            body = "；".join(dict.fromkeys(reversed(current_round)))   # 去重但保序
            return AIMessage(content=f"根据工具返回的结果：{body}")

        # 情况 2：用户提问，且身上绑了工具 → 试着挑一个来调
        text = last.content if isinstance(last.content, str) else str(last.content)
        picked = self._pick_tool(text)
        if picked is not None:
            name, args = picked
            return AIMessage(
                content="",
                # tool_calls 是 LangChain 约定的结构，四个字段缺一不可：
                #   name 工具名 / args 参数字典 / id 唯一标识 / type 固定值
                tool_calls=[{"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}],
            )

        # 情况 3：闲聊
        # 这里故意把"我能看到多少条历史"打印出来 —— 因为在真模型下，
        # 记忆的效果体现在回答内容里；换成假模型就看不出差别了，
        # 只能靠这个计数让你确认 checkpointer 到底有没有在干活。
        turns = sum(1 for m in messages if m.type == "human")
        return AIMessage(
            content=(
                f"（离线模式）这是本次会话的第 {turns} 轮，"
                f"会话里现在一共 {len(messages)} 条消息。"
                f"你说的「{text}」在配好 API Key 后会由真模型回答。"
            )
        )

    def _pick_tool(self, text: str) -> tuple[str, dict[str, Any]] | None:
        """按 description 的字面重合度挑一个工具，重合度为 0 就不挑。"""
        best_tool, best_score = None, 0
        for tool_obj in self.bound_tools:
            desc = getattr(tool_obj, "description", "") or ""
            score = _overlap_score(desc, text)
            if score > best_score:
                best_tool, best_score = tool_obj, score
        if best_tool is None:
            return None

        # 参数怎么填？取 schema 里第一个必填字段，值直接塞用户原话。
        # 真模型会自己抽参数，这里只是让流程能跑通 —— 具体清洗交给工具自己。
        _, required = _schema_parts(best_tool)
        param = required[0] if required else None
        args = {param: text} if param else {}
        return getattr(best_tool, "name", "unknown"), args

    @staticmethod
    def _fake_args(props: dict[str, Any]) -> dict[str, Any]:
        """照着 JSON Schema 的 properties 编一份类型正确的假数据，好让 Pydantic 校验能通过。"""
        out: dict[str, Any] = {}
        for field_name, spec in props.items():
            kind = (spec or {}).get("type", "string")
            if kind in ("integer", "number"):
                out[field_name] = 0
            elif kind == "boolean":
                out[field_name] = False
            elif kind == "array":
                out[field_name] = []
            elif kind == "object":
                out[field_name] = {}
            else:
                out[field_name] = "（离线模式的示例值）"
        return out
