"""
config.py —— 模型工厂：全项目唯一决定「用哪个大模型」的地方。

【为什么单独抽一个文件】
    LangChain 最容易犯的错是"到处 new 一个 ChatOpenAI"。
    等你哪天要从 GPT 换成 DeepSeek、或者换成本地 Ollama，就得改十几个文件。
    把「造模型」这件事收敛到一个函数里，以后换模型只改 .env，业务代码一行不动。

【在 LangChain 里的位置】
    LangChain 的所有能力（prompt / tool / agent / parser）都只认一个东西：
    BaseChatModel 接口。谁实现这个接口，谁就能接进来。
    所以"换模型"本质是换一个实现了 BaseChatModel 的实例，别的都不用动。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

# ---------------------------------------------------------------------------
# 1. 先加载 .env
# ---------------------------------------------------------------------------
# load_dotenv 会去读文件、把里面的 KEY=VALUE 塞进 os.environ。
# override=False 表示「系统环境变量优先」，避免 .env 里写了个旧 key 把真的盖掉。
# 文件不存在也不会报错 —— 新项目克隆下来没配 key 时，这里不能拦住人。
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


# ---------------------------------------------------------------------------
# 2. 国内常见的「OpenAI 兼容」平台
# ---------------------------------------------------------------------------
# 这些平台都实现了 /v1/chat/completions 协议，所以不必各装一个 SDK，
# 直接用 langchain-openai 的 ChatOpenAI 改个 base_url 就能对接。
# 格式：(接口地址, 默认模型名)
COMPATIBLE_PROVIDERS: dict[str, tuple[str, str]] = {
    "openai":    ("https://api.openai.com/v1", "gpt-4o-mini"),
    "deepseek":  ("https://api.deepseek.com/v1", "deepseek-chat"),
    "dashscope": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
    "moonshot":  ("https://api.moonshot.cn/v1", "moonshot-v1-8k"),
}


def _env_key(provider: str) -> str:
    """约定：DEEPSEEK 的 key 就叫 DEEPSEEK_API_KEY。"""
    return f"{provider.upper()}_API_KEY"


def _guess_provider() -> str:
    """没显式配置时，看环境里哪个 key 存在就用哪个；一个都没有就走离线模式。"""
    forced = (os.getenv("LC_PROVIDER") or "").strip().lower()
    if forced:
        return forced
    for name in COMPATIBLE_PROVIDERS:
        if os.getenv(_env_key(name)):
            return name
    return "offline"   # ← 没有任何 key 时的兜底：本地假模型，不需要联网


def get_model(temperature: float = 0.0, model: str | None = None, **overrides) -> BaseChatModel:
    """
    返回一个可直接用的聊天模型实例。

    参数：
        temperature: 0 = 尽量稳定（写代码、抽数据用）；0.7 = 发散（写文案用）
        model:       手动指定模型名，覆盖默认值
        **overrides: 透传给底层模型构造函数的其它参数（如 max_tokens）

    【为什么统一返回 BaseChatModel】
        因为下面所有文件（prompt、agent、parser）都只依赖这个抽象接口，
        调用方永远不需要知道背后到底是 OpenAI 还是本地假模型。
    """
    provider = _guess_provider()

    # --- 兜底：离线模式 ---------------------------------------------------
    if provider == "offline":
        # 延迟导入：只有真的走离线才去加载这个模块，避免无谓的依赖
        from offline_model import OfflineChatModel

        return OfflineChatModel(temperature=temperature, **overrides)

    # --- 真实模型 ---------------------------------------------------------
    if provider not in COMPATIBLE_PROVIDERS:
        raise ValueError(
            f"不认识的 provider: {provider!r}，可选：{list(COMPATIBLE_PROVIDERS)} 或 offline"
        )

    base_url, default_model = COMPATIBLE_PROVIDERS[provider]
    api_key = os.getenv(_env_key(provider))

    # init_chat_model 是 LangChain 1.x 的官方统一入口：
    # 它按 model_provider 去挑对应的集成包，省掉一堆 if/else + import。
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model=model or os.getenv("LC_MODEL") or default_model,
        model_provider="openai",      # 兼容协议统一走 openai 适配器
        base_url=base_url,            # 只改地址，不改代码
        api_key=api_key,
        temperature=temperature,
        **overrides,
    )


def describe() -> str:
    """打印当前会走哪条路，排查「为什么我的 key 没生效」时很有用。"""
    provider = _guess_provider()
    if provider == "offline":
        return "当前模型：离线假模型（没有检测到任何 *_API_KEY，功能可用但没有真实智能）"
    base_url, default_model = COMPATIBLE_PROVIDERS[provider]
    model = os.getenv("LC_MODEL") or default_model
    key = os.getenv(_env_key(provider)) or ""
    masked = f"{key[:6]}...{key[-4:]}" if len(key) > 12 else "(空)"
    return f"当前模型：{provider} / {model} @ {base_url}，key={masked}"


if __name__ == "__main__":
    print(describe())
