"""
模型池 + 预热机制
提供全局单例 LLM 实例池，避免重复创建，加速推理
支持分级推理：大模型(7B)处理复杂任务，小模型(0.5B)处理简单任务
"""
import time
from typing import Optional, Dict
from .local_qwen import LocalQwenLLM, get_local_model, get_local_model_by_size


class ModelPool:
    """全局模型池（单例）"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not ModelPool._initialized:
            print("[ModelPool] 🚀 初始化模型池（精简版：只加载必需模型）...")
            
            # 🔥 不再加载14B模型（ComfortTool 改用 API）
            
            # 预加载大模型（7B，所有本地任务）
            start_time = time.time()
            model_7b, tokenizer_7b = get_local_model_by_size("7b-gptq")
            load_time_7b = time.time() - start_time
            print(f"[ModelPool] ✅ 7B模型已加载 (耗时 {load_time_7b:.2f}秒)")
            
            # 🔥 不再加载0.5B模型（改用7B-GPTQ替代）
            
            # 创建不同温度的 LLM 实例池
            self._llm_pool: Dict[str, LocalQwenLLM] = {}
            
            # === 7B大模型实例（只创建实际需要的 3 个） ===
            print("[ModelPool] 📦 创建7B实例（最小化）:")
            self._create_llm_instance('eval_long', model_size="7b-gptq", temperature=0.3, max_new_tokens=32)      # AnswerEvaluationTool
            self._create_llm_instance('small_classify', model_size="7b-gptq", temperature=0.2, max_new_tokens=48) # 打断检测、话题映射
            self._create_llm_instance('7b_complex', model_size="7b-gptq", temperature=0.7, max_new_tokens=256)    # StandardQuestionTool, 话题选择
            
            # 别名映射（不重复创建实例）
            self._llm_pool['default'] = self._llm_pool['7b_complex']
            self._llm_pool['7b_default'] = self._llm_pool['7b_complex']
            self._llm_pool['precise'] = self._llm_pool['eval_long']
            
            print(f"[ModelPool] 📊 共预创建 {3} 个 LLM 实例 + {3} 个别名")
            
            # 预热模型（首次推理最慢，预热后加速）
            self._warmup()
            
            ModelPool._initialized = True
            print("[ModelPool] ✅ 模型池初始化完成\n")
    
    def _create_llm_instance(self, name: str, model_size: str, temperature: float, max_new_tokens: int):
        """创建并缓存 LLM 实例"""
        llm = LocalQwenLLM(
            model_size=model_size,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
        self._llm_pool[name] = llm
        print(f"[ModelPool]   - {name}: model={model_size.upper()}, temp={temperature}, max_tokens={max_new_tokens}")
    
    def _warmup(self):
        """预热模型（只预热1次，所有实例共享同一个7B底层模型）"""
        print("[ModelPool] 🔥 预热 7B 模型（仅1次推理）...")
        warmup_prompt = "你好"
        
        try:
            start_time = time.time()
            # 所有实例共享同一个 7B-GPTQ 底层模型，只需预热一次
            _ = self._llm_pool['small_classify'].invoke(warmup_prompt)
            warmup_time = time.time() - start_time
            print(f"[ModelPool] ✅ 预热完成 (耗时 {warmup_time:.2f}秒)")
        except Exception as e:
            print(f"[ModelPool] ⚠️ 预热失败: {type(e).__name__}")
        
        print(f"[ModelPool] 💡 后续推理将快 2-3 倍\n")
    
    def get_llm(
        self, 
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        pool_key: Optional[str] = None
    ) -> LocalQwenLLM:
        """
        获取 LLM 实例
        
        Args:
            temperature: 温度参数（如果指定，会匹配最接近的池化实例）
            max_new_tokens: 最大token数
            pool_key: 直接指定池中的key（'default', 'precise', 'creative', 'agent'）
        
        Returns:
            LocalQwenLLM 实例
        """
        # 优先使用 pool_key
        if pool_key and pool_key in self._llm_pool:
            return self._llm_pool[pool_key]
        
        # 如果没有指定任何参数，返回默认
        if temperature is None and max_new_tokens is None:
            return self._llm_pool['default']
        
        # 根据温度匹配最接近的实例
        if temperature is not None:
            if temperature <= 0.4:
                return self._llm_pool['precise']
            elif temperature >= 0.8:
                return self._llm_pool['creative']
            else:
                return self._llm_pool['default']
        
        # 如果都不匹配，返回默认
        return self._llm_pool['default']
    
    def get_stats(self) -> dict:
        """获取模型池统计信息"""
        return {
            'pool_size': len(self._llm_pool),
            'available_keys': list(self._llm_pool.keys()),
            'initialized': ModelPool._initialized
        }


# 全局单例实例
_model_pool_instance: Optional[ModelPool] = None


def get_model_pool() -> ModelPool:
    """获取全局模型池实例"""
    global _model_pool_instance
    if _model_pool_instance is None:
        _model_pool_instance = ModelPool()
    return _model_pool_instance


def get_pooled_llm(
    temperature: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    pool_key: Optional[str] = None
) -> LocalQwenLLM:
    """
    便捷函数：从模型池获取 LLM 实例
    
    Examples:
        >>> llm = get_pooled_llm()  # 默认实例
        >>> llm = get_pooled_llm(pool_key='agent')  # Agent专用
        >>> llm = get_pooled_llm(temperature=0.3)  # 自动匹配 precise
    """
    pool = get_model_pool()
    return pool.get_llm(temperature, max_new_tokens, pool_key)


if __name__ == "__main__":
    # 测试
    print("=== 测试模型池 ===\n")
    
    # 初始化模型池
    pool = get_model_pool()
    
    # 获取统计信息
    print("\n模型池信息:")
    print(pool.get_stats())
    
    # 测试获取不同实例
    print("\n测试获取实例:")
    llm1 = get_pooled_llm(pool_key='default')
    llm2 = get_pooled_llm(temperature=0.3)
    llm3 = get_pooled_llm(pool_key='agent')
    
    print(f"default == precise(temp=0.3): {llm1 is llm2}")
    print(f"precise(temp=0.3) == agent: {llm2 is llm3}")
    
    # 测试推理
    print("\n测试推理:")
    start = time.time()
    response = llm1.invoke("1+1等于多少？")
    print(f"回复: text_chars={len(str(response))}")
    print(f"耗时: {time.time() - start:.2f}秒")
