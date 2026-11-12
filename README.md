# LangChain / LangGraph 学习项目

渐进式学习的代码仓库。每个文件只讲一个概念，都能独立跑，改坏一个不影响别的。

## 从哪开始看

**先看 `lc/05_agent.py`** —— 那是能用的东西。跑一遍，看到 agent 怎么自己调工具、怎么记住上下文。
其余章节是它的零件，遇到看不懂的再回头翻。

```bash
python main.py 5                    # 跑 agent 的固定演示
python lc/05_agent.py chat          # 进入交互模式，真的跟它聊
python main.py                      # 列出所有章节
```

## 三步跑起来

```bash
# 1. 装依赖（venv 已经有了）
uv pip install --python .venv/Scripts/python.exe -e .

# 2. 配 key（可选，不配也能跑，走离线假模型）
cp .env.example .env
# 编辑 .env，填 DEEPSEEK_API_KEY

# 3. 跑
./.venv/Scripts/python.exe lc/05_agent.py
```

没配 key 也能全流程跑通，因为 `lc/offline_model.py` 提供了一个实现 `BaseChatModel` 接口的本地假模型。
它的输出是假的，但**协议是真的** —— agent 循环、工具调用、流式输出全都会真实执行一遍。
配好 key 之后，业务代码一个字都不用改，`config.get_model()` 会自动切到真模型。

## 目录

```
lc/
├── config.py          模型工厂：全项目唯一决定"用哪个模型"的地方
├── offline_model.py   离线假模型：没有 API Key 也能跑
├── 01_messages.py     消息类型（对话的本质是消息列表）
├── 02_prompt_lcel.py  提示词模板 + LCEL 管道
├── 03_parsers.py      输出解析器 + 结构化输出
├── 04_tools.py        工具定义（模型只"申请"，不执行）
├── 05_agent.py        ★ agent：工具循环 + 记忆 + 轨迹
├── 06_stream.py       流式输出
└── 07_middleware.py   中间件（1.x 新特性）
main.py                总入口
学习笔记.md             每次对话的总结，按时间往下加
```

## 版本

| 包 | 版本 |
|---|---|
| langchain | 1.4.9 |
| langchain-core | 1.7.7 |
| langchain-openai | 1.7.4 |
| langgraph | 1.2.20 |

LangChain 1.x 跟网上大量 0.1/0.2 的教程**不兼容**：
`initialize_agent`、`AgentExecutor`、`LLMChain` 这些都已经进历史博物馆了。
现在只有一个入口 `create_agent()`，底层是 LangGraph 的状态图。
