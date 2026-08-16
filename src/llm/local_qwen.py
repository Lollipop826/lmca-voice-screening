"""
本地 Qwen2.5-7B 模型服务
使用 transformers 加载，提供 LangChain 兼容接口
"""
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Any
import os
import threading
from pathlib import Path

# 模型路径配置（支持多模型）
_PROJECT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATHS = {
    "7b": os.getenv(
        "QWEN_7B_MODEL_PATH",
        str(_PROJECT_MODELS_DIR / "Qwen2.5-7B-Instruct-GPTQ-Int4"),
    ),
    "7b-gptq": os.getenv(
        "QWEN_7B_MODEL_PATH",
        str(_PROJECT_MODELS_DIR / "Qwen2.5-7B-Instruct-GPTQ-Int4"),
    ),
    "0.5b": os.getenv(
        "QWEN_05B_MODEL_PATH",
        str(_PROJECT_MODELS_DIR / "Qwen2.5-0.5B-Instruct"),
    ),
    "14b": os.getenv(
        "QWEN_14B_MODEL_PATH",
        str(_PROJECT_MODELS_DIR / "Qwen2.5-14B-Instruct-GPTQ-Int4"),
    ),
    "14b-gptq": os.getenv(
        "QWEN_14B_MODEL_PATH",
        str(_PROJECT_MODELS_DIR / "Qwen2.5-14B-Instruct-GPTQ-Int4"),
    ),
}

# 默认模型
MODEL_PATH = MODEL_PATHS["7b"]

# 全局模型实例（支持多模型单例）
_models = {}  # {model_size: (model, tokenizer)}
_model = None  # 向后兼容
_tokenizer = None  # 向后兼容

# 🔒 线程锁：防止并发访问同一模型导致 CUDA 错误
_model_locks = {}  # {model_size: threading.Lock()}
_global_generate_lock = threading.Lock()


def get_local_model_by_size(model_size: str = "7b"):
    """
    按大小获取本地模型单例
    
    Args:
        model_size: 模型大小，可选 "7b", "0.5b", "14b"
    
    Returns:
        (model, tokenizer) 元组
    """
    global _models
    
    if model_size not in _models:
        if model_size not in MODEL_PATHS:
            raise ValueError(f"不支持的模型大小: {model_size}，可选: {list(MODEL_PATHS.keys())}")
        
        model_path = MODEL_PATHS[model_size]
        model_name = model_path.split('/')[-1]
        model_name = model_path.split('/')[-1]
        # print(f"[LocalQwen] 正在加载模型 ({model_size.upper()}): {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 检测是否是 GPTQ 量化模型
        is_gptq = "GPTQ" in model_path or "gptq" in model_path
        
        if is_gptq:
            # GPTQ 量化模型：自动识别量化配置
            # print(f"[LocalQwen] 检测到 GPTQ 量化模型，使用优化加载...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",  # GPTQ 会自动处理
                trust_remote_code=True
            )
        else:
            # 普通 FP16 模型
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        model.eval()
        _models[model_size] = (model, tokenizer)
        
        # 🔒 为每个模型创建独立的线程锁
        _model_locks[model_size] = threading.Lock()
        
        # print(f"[LocalQwen] ✅ 模型加载完成 ({model_size.upper()}): {model_name}")
    
    return _models[model_size]


def get_local_model():
    """获取默认本地模型单例（向后兼容）"""
    global _model, _tokenizer
    
    if _model is None:
        _model, _tokenizer = get_local_model_by_size("7b")
    
    return _model, _tokenizer


class LocalQwenLLM(LLM):
    """LangChain 兼容的本地 Qwen LLM"""
    
    model_path: str = MODEL_PATH
    model_size: str = "7b"  # 支持选择模型大小
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    @property
    def _llm_type(self) -> str:
        return "local_qwen"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        model, tokenizer = get_local_model_by_size(self.model_size)
        
        # 构建消息格式
        messages = [{"role": "user", "content": prompt}]
        
        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码（加入长度保护，避免超过模型最大上下文导致 CUDA index out of bounds）
        model_max_ctx = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        tok_max_ctx = getattr(tokenizer, "model_max_length", None)
        max_ctx = None
        for v in (model_max_ctx, tok_max_ctx):
            if isinstance(v, int) and v > 0 and v < 10**9:
                max_ctx = v if max_ctx is None else min(max_ctx, v)
        if max_ctx is None:
            max_ctx = 8192
        reserved = min(max(64, self.max_new_tokens + 32), max_ctx - 1)
        max_input_tokens = max(1, max_ctx - reserved)

        inputs = tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        ).to(model.device)
        
        # ReAct 格式的停止词
        stop_sequences = stop or []
        stop_sequences.extend(["\nObservation:", "\nObservation :", "Observation:"])
        
        # 生成
        from transformers import StoppingCriteria, StoppingCriteriaList
        
        class StopOnTokens(StoppingCriteria):
            def __init__(self, stop_ids):
                self.stop_ids = stop_ids
            def __call__(self, input_ids, scores, **kwargs):
                for stop_id in self.stop_ids:
                    if input_ids[0][-len(stop_id):].tolist() == stop_id:
                        return True
                return False
        
        # 编码停止词
        stop_ids = []
        for seq in stop_sequences:
            ids = tokenizer.encode(seq, add_special_tokens=False)
            if ids:
                stop_ids.append(ids)
        
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)]) if stop_ids else None
        
        # 🛡️ 数值稳定性保护：防止 probability tensor 出现 NaN/Inf
        safe_temperature = max(0.1, min(self.temperature, 2.0))  # 限制在 [0.1, 2.0]
        safe_top_p = max(0.1, min(self.top_p, 0.95))  # 限制 top_p 避免极端值
        
        # 🔒 使用线程锁防止并发 GPU 访问（修复 CUDA 错误）
        model_lock = _model_locks.get(self.model_size)
        if model_lock is None:
            model_lock = threading.Lock()
            _model_locks[self.model_size] = model_lock
        
        from transformers import LogitsProcessor, LogitsProcessorList
        
        class StabilizeLogitsProcessor(LogitsProcessor):
            def __call__(self, input_ids, scores):
                # 检测并修复 NaN/Inf
                if torch.isnan(scores).any() or torch.isinf(scores).any():
                    # print(f"[LocalQwen] 🔧 检测到并修复 NaN/Inf logits")
                    # 将 NaN/Inf 替换为极小值
                    scores = torch.where(torch.isnan(scores) | torch.isinf(scores), 
                                       torch.full_like(scores, -1e10), scores)
                
                # 确保 logits 在合理范围内
                scores = torch.clamp(scores, min=-50.0, max=50.0)
                
                # 数值稳定性：减去最大值防止溢出
                scores = scores - scores.max(dim=-1, keepdim=True)[0]
                
                return scores

        logits_processor = LogitsProcessorList([StabilizeLogitsProcessor()])

        # 🔥 性能优化：移除 _global_generate_lock，只保留 per-model lock
        # 原来的双层锁让所有本地推理串行化，API+Local 无法并行
        # per-model lock 已足够保证同一模型的 CUDA 安全
        with model_lock:
            with torch.no_grad():
                try:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=max(0.3, min(safe_temperature, 1.0)),
                        top_p=max(0.3, min(safe_top_p, 0.9)),
                        top_k=20,
                        do_sample=True,
                        repetition_penalty=1.02,
                        pad_token_id=tokenizer.eos_token_id,
                        stopping_criteria=stopping_criteria,
                        logits_processor=logits_processor,
                    )
                except RuntimeError as e:
                    if "probability tensor" in str(e):
                        print(f"[LocalQwen] ⚠️ 稳定化生成失败，降级到贪婪解码: {type(e).__name__}")
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=min(self.max_new_tokens, 200),
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                            stopping_criteria=stopping_criteria,
                            logits_processor=logits_processor,
                        )
                    else:
                        raise e
        
        # 解码
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 后处理：确保只返回一个完整的思考步骤
        response = response.strip()
        
        # 如果同时包含 Action 和 Final Answer，只保留第一个
        if "Final Answer:" in response and "Action:" in response:
            fa_pos = response.find("Final Answer:")
            ac_pos = response.find("Action:")
            if ac_pos < fa_pos:
                # Action 在前，截取到 Final Answer 之前
                response = response[:fa_pos].strip()
            else:
                # Final Answer 在前，截取到 Action 之前
                response = response[:ac_pos].strip()
        
        return response


# 便捷函数
def create_local_llm(**kwargs) -> LocalQwenLLM:
    """创建本地 LLM 实例"""
    return LocalQwenLLM(**kwargs)


if __name__ == "__main__":
    # 测试
    llm = create_local_llm()
    response = llm.invoke("你好，请介绍一下自己")
    print(f"回复: text_chars={len(str(response))}")
