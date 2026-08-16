"""
共享 HTTP 客户端池

优化 API 调用性能：
1. 单例模式共享客户端
2. 增大连接池大小
3. 启用 Keep-Alive
"""
import httpx
from typing import Optional, Any, Dict
from functools import lru_cache


# 全局共享的 httpx 客户端配置
_shared_async_client: Optional[httpx.AsyncClient] = None
_shared_sync_client: Optional[httpx.Client] = None

# 🔥 全局模型切换：doubao-seed-2-0-mini-260215 / doubao-seed-2-0-lite-260215
_ARK_MODEL_OPTIONS = ["doubao-seed-2-0-mini-260215", "doubao-seed-2-0-lite-260215"]
_active_ark_model: Optional[str] = None  # None = 使用环境变量默认值


def get_active_ark_model() -> str:
    """获取当前激活的 ARK 模型名"""
    import os
    return _active_ark_model or os.getenv("TOPIC_SELECTION_MODEL", "doubao-seed-2-0-mini-260215")


def switch_ark_model(model_name: str) -> str:
    """
    切换全局 ARK 模型，清除 LRU 缓存使新模型生效。
    返回实际切换到的模型名。
    """
    global _active_ark_model
    if model_name not in _ARK_MODEL_OPTIONS:
        raise ValueError(f"不支持的模型: {model_name}, 可选: {_ARK_MODEL_OPTIONS}")
    _active_ark_model = model_name
    # 清除 LRU 缓存，下次调用会用新模型重建实例
    get_volcengine_chat_openai.cache_clear()
    get_siliconflow_chat_openai.cache_clear()
    get_siliconflow_chatmodel.cache_clear()
    print(f"[HTTPClient] 🔄 全局模型已切换为: {model_name}")
    return model_name


def get_shared_httpx_client() -> httpx.Client:
    """
    获取共享的同步 httpx 客户端
    
    - 连接池大小: 100 (默认10)
    - Keep-Alive 超时: 60s
    - 连接超时: 10s
    """
    global _shared_sync_client
    
    if _shared_sync_client is None:
        # 创建带大连接池的 httpx 客户端
        limits = httpx.Limits(
            max_keepalive_connections=100,  # Keep-Alive 连接数（默认20）
            max_connections=200,            # 最大连接数（默认100）
            keepalive_expiry=60.0,          # Keep-Alive 超时（秒）
        )
        
        _shared_sync_client = httpx.Client(
            limits=limits,
            timeout=httpx.Timeout(
                connect=10.0,   # 连接超时
                read=60.0,      # 读取超时
                write=30.0,     # 写入超时
                pool=10.0,      # 连接池等待超时
            ),
            http2=False,  # 禁用 HTTP/2（需要额外安装 h2 包）
        )
        print("[HTTPClient] ✅ 创建共享 HTTP 客户端池 (max=200, keepalive=100)")
    
    return _shared_sync_client


def get_shared_async_httpx_client() -> httpx.AsyncClient:
    """
    获取共享的异步 httpx 客户端
    """
    global _shared_async_client
    
    if _shared_async_client is None:
        limits = httpx.Limits(
            max_keepalive_connections=100,
            max_connections=200,
            keepalive_expiry=60.0,
        )
        
        _shared_async_client = httpx.AsyncClient(
            limits=limits,
            timeout=httpx.Timeout(
                connect=10.0,
                read=60.0,
                write=30.0,
                pool=10.0,
            ),
            http2=False,  # 禁用 HTTP/2
        )
        print("[HTTPClient] ✅ 创建共享异步 HTTP 客户端池")
    
    return _shared_async_client


@lru_cache(maxsize=1)
def get_siliconflow_chatmodel():
    """
    获取共享的 SiliconFlow ChatOpenAI 实例
    
    使用单例模式，所有工具共享同一个 LLM 客户端
    """
    # 向后兼容：原先固定返回 72B 的单例
    return get_siliconflow_chat_openai(
        model="Qwen/Qwen2.5-72B-Instruct",
        temperature=0.7,
        max_tokens=200,
        timeout=30,
    )


@lru_cache(maxsize=32)
def get_siliconflow_chat_openai(
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: Any = 30,
    max_retries: int = 1,
    streaming: bool = False,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    disable_thinking: bool = False,
):
    """
    获取共享的 SiliconFlow ChatOpenAI 实例（按配置缓存）。
    
    目的：
    - 复用同一个 ChatOpenAI 对象（避免重复初始化）
    - 复用同一个 httpx.Client / httpx.AsyncClient（连接池 + Keep-Alive）
    
    注意：
    - 不同参数（model/temperature/max_tokens/timeout/streaming/disable_thinking）会生成不同缓存实例
    - disable_thinking=True 会关闭 Qwen3 等模型的思考模式（大幅降低延迟）
    """
    import os
    from langchain_openai import ChatOpenAI

    http_client = get_shared_httpx_client()
    http_async_client = get_shared_async_httpx_client()

    _base_url = base_url or os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    _api_key = api_key or os.getenv("SILICONFLOW_API_KEY")

    kwargs = dict(
        model=model,
        base_url=_base_url,
        api_key=_api_key,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        streaming=streaming,
        http_client=http_client,
        http_async_client=http_async_client,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if disable_thinking:
        kwargs["extra_body"] = {"enable_thinking": False}

    llm = ChatOpenAI(**kwargs)
    thinking_info = " thinking=OFF" if disable_thinking else ""
    print(
        f"[HTTPClient] ✅ 创建共享 ChatOpenAI: model={model} temp={temperature} "
        f"max_tokens={max_tokens} streaming={streaming} timeout={timeout}{thinking_info}"
    )
    return llm


@lru_cache(maxsize=32)
def get_volcengine_chat_openai(
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: Any = 30,
    max_retries: int = 1,
    streaming: bool = False,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    disable_thinking: bool = True,
):
    """
    获取共享的火山引擎 ChatOpenAI 实例（按配置缓存）。
    
    火山引擎 Doubao 模型默认开启深度思考，必须显式关闭，否则 content 为空。
    关闭方式：extra_body={"thinking": {"type": "disabled"}}
    """
    import os
    from langchain_openai import ChatOpenAI

    http_client = get_shared_httpx_client()
    http_async_client = get_shared_async_httpx_client()

    _base_url = base_url or os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    _api_key = api_key or os.getenv("ARK_API_KEY")

    kwargs = dict(
        model=model,
        base_url=_base_url,
        api_key=_api_key,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        streaming=streaming,
        http_client=http_client,
        http_async_client=http_async_client,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if disable_thinking:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    llm = ChatOpenAI(**kwargs)
    thinking_info = " thinking=OFF" if disable_thinking else ""
    print(
        f"[HTTPClient] ✅ 创建火山引擎 ChatOpenAI: model={model} temp={temperature} "
        f"max_tokens={max_tokens} streaming={streaming} timeout={timeout}{thinking_info}"
    )
    return llm


def get_dashscope_chat_openai(
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: Any = 30,
    max_retries: int = 1,
    streaming: bool = False,
    disable_thinking: bool = False,
):
    """Create a DashScope OpenAI-compatible chat client using the shared pool."""
    import os

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("未配置 DASHSCOPE_API_KEY")
    return get_siliconflow_chat_openai(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        streaming=streaming,
        base_url=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        api_key=api_key,
        disable_thinking=disable_thinking,
    )


def get_chat_openai(
    model: str = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: Any = 30,
    max_retries: int = 1,
    streaming: bool = False,
    disable_thinking: bool = True,
):
    """
    自动选择 LLM 提供商：
    - 有 DASHSCOPE_API_KEY → 阿里云百炼，默认 qwen3.7-flash
    - 否则有 ARK_API_KEY → 火山引擎 (Doubao)
    - 否则 → SiliconFlow (Qwen)
    """
    import os
    if os.getenv("DASHSCOPE_API_KEY"):
        _model = model or os.getenv("DASHSCOPE_CHAT_MODEL", "qwen3.7-flash")
        return get_dashscope_chat_openai(
            model=_model, temperature=temperature, max_tokens=max_tokens,
            timeout=timeout, max_retries=max_retries, streaming=streaming,
            disable_thinking=disable_thinking,
        )
    if os.getenv("ARK_API_KEY"):
        _model = model or get_active_ark_model()
        return get_volcengine_chat_openai(
            model=_model, temperature=temperature, max_tokens=max_tokens,
            timeout=timeout, max_retries=max_retries, streaming=streaming,
            disable_thinking=disable_thinking,
        )
    else:
        _model = model or os.getenv("TOPIC_SELECTION_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
        return get_siliconflow_chat_openai(
            model=_model, temperature=temperature, max_tokens=max_tokens,
            timeout=timeout, max_retries=max_retries, streaming=streaming,
            disable_thinking=disable_thinking,
        )


# ============================================================
# 🔥 VolcEngine Context Cache API（缓存 system prompt 减少延迟和 token 消耗）
# ============================================================
_context_cache_ids: Dict[str, str] = {}  # cache_key → context_id


def create_volcengine_context_cache(
    model: str,
    system_prompt: str,
    mode: str = "session",
    ttl: int = 3600,
) -> Optional[str]:
    """
    创建火山引擎上下文缓存，返回 context_id。
    
    - common_prefix 模式：缓存 system prompt，后续调用自动复用
    - 相同 model + system_prompt 不会重复创建
    - 返回 None 表示创建失败（不影响正常调用，降级为普通模式）
    """
    import os
    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        return None

    # 环境变量开关：默认关闭（doubao-seed 系列不支持 Context Cache）
    if os.getenv("USE_CONTEXT_CACHE", "false").lower() != "true":
        return None

    # Context Cache 只对推理接入点（ep-xxx）有效，普通模型名直接跳过
    if not model.startswith("ep-"):
        return None

    cache_key = f"{model}:{hash(system_prompt)}"
    if cache_key in _context_cache_ids:
        print(f"[ContextCache] ♻️ 复用已有缓存: {_context_cache_ids[cache_key]}")
        return _context_cache_ids[cache_key]

    base_url = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    url = f"{base_url}/context/create"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}],
        "mode": mode,
        "ttl": ttl,
    }

    try:
        client = get_shared_httpx_client()
        resp = client.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        context_id = data.get("id")
        if context_id:
            _context_cache_ids[cache_key] = context_id
            cached_tokens = data.get("usage", {}).get("prompt_tokens", 0)
            print(
                f"[ContextCache] ✅ 创建成功: model={model}, "
                f"context_id={context_id}, cached_tokens={cached_tokens}, "
                f"mode={mode}, ttl={ttl}s"
            )
            return context_id
        else:
            print(f"[ContextCache] ⚠️ 响应中无 id: field_count={len(data) if isinstance(data, dict) else 0}")
            return None
    except Exception as e:
        print(f"[ContextCache] ❌ 创建失败（降级为普通模式）: {type(e).__name__}")
        return None


def get_volcengine_context_chat_openai(
    context_id: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: Any = 30,
    max_retries: int = 1,
    streaming: bool = False,
) -> 'ChatOpenAI':
    """
    获取使用上下文缓存的火山引擎 ChatOpenAI 实例。
    
    base_url 指向 /api/v3/context，LangChain 自动拼接 /chat/completions
    → 最终请求 /api/v3/context/chat/completions
    
    extra_body 中携带 context_id + thinking disabled。
    调用时无需再传 system message。
    """
    import os
    from langchain_openai import ChatOpenAI

    http_client = get_shared_httpx_client()
    http_async_client = get_shared_async_httpx_client()

    base_url = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    context_base_url = f"{base_url}/context"
    api_key = os.getenv("ARK_API_KEY")

    kwargs = dict(
        model=model,
        base_url=context_base_url,
        api_key=api_key,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        streaming=streaming,
        http_client=http_client,
        http_async_client=http_async_client,
        extra_body={
            "context_id": context_id,
            "thinking": {"type": "disabled"},
        },
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    llm = ChatOpenAI(**kwargs)
    print(
        f"[ContextCache] ✅ 创建 Context ChatOpenAI: model={model} "
        f"context_id={context_id} temp={temperature} max_tokens={max_tokens}"
    )
    return llm


def cleanup_clients():
    """清理客户端资源（应用关闭时调用）"""
    global _shared_sync_client, _shared_async_client
    
    if _shared_sync_client:
        _shared_sync_client.close()
        _shared_sync_client = None
        
    if _shared_async_client:
        import asyncio
        try:
            asyncio.get_event_loop().run_until_complete(_shared_async_client.aclose())
        except:
            pass
        _shared_async_client = None
    
    print("[HTTPClient] 🔒 已清理共享客户端")
