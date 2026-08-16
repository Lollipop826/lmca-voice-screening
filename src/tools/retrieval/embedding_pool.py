"""
Embedding 模型池 - 单例模式，避免重复加载
"""
import os
import time
from pathlib import Path
from typing import Optional
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    HuggingFaceEmbeddings = None


class EmbeddingPool:
    """全局 Embedding 模型池（单例）"""
    
    _instance = None
    _initialized = False
    _embedding_model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not EmbeddingPool._initialized:
            print("[EmbeddingPool] 🚀 初始化 Embedding 模型池...")
            
            # 默认使用 BGE-M3 模型；优先读取环境变量 EMBEDDING_MODEL_PATH，否则用本地路径
            default_model_path = (
                Path(__file__).resolve().parents[3]
                / "models"
                / "bge-m3"
            )
            self.model_path = os.getenv(
                "EMBEDDING_MODEL_PATH",
                str(default_model_path),
            )
            self.device = os.getenv("EMBEDDING_DEVICE", "cuda")
            
            # 预加载模型
            start_time = time.time()
            self._load_embedding_model()
            load_time = time.time() - start_time
            
            print(f"[EmbeddingPool] ✅ Embedding 模型已加载 (耗时 {load_time:.2f}秒)")
            
            # 预热：编码一个测试句子
            self._warmup()
            
            EmbeddingPool._initialized = True
            print("[EmbeddingPool] ✅ 模型池初始化完成\n")
    
    def _load_embedding_model(self):
        """加载 Embedding 模型"""
        EmbeddingPool._embedding_model = HuggingFaceEmbeddings(
            model_name=self.model_path,
            # 使用本地模型文件，不从网上下载
            model_kwargs={"device": self.device, "local_files_only": True},
            encode_kwargs={"normalize_embeddings": True}
        )
    
    def _warmup(self):
        """预热模型（首次推理最慢）"""
        print("[EmbeddingPool] 🔥 开始预热 Embedding 模型...")
        start_time = time.time()
        
        try:
            # 编码测试句子
            test_text = "阿尔茨海默病认知评估"
            _ = EmbeddingPool._embedding_model.embed_query(test_text)
            
            warmup_time = time.time() - start_time
            print(f"[EmbeddingPool] ✅ 预热完成 (耗时 {warmup_time:.2f}秒)")
            print(f"[EmbeddingPool] 💡 后续 Embedding 将快 2-3 倍\n")
        except Exception as e:
            print(f"[EmbeddingPool] ⚠️ 预热失败（不影响使用）: {type(e).__name__}\n")
    
    def get_embeddings(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None
    ) -> HuggingFaceEmbeddings:
        """
        获取 Embedding 模型实例
        
        Args:
            model_path: 模型路径（如果为None，使用默认BGE-M3）
            device: 设备（cuda/cpu）
        
        Returns:
            HuggingFaceEmbeddings 实例
        """
        # 如果参数与默认一致，返回池化实例
        # 兼容：外部可能传入 HuggingFace repo id（如 "BAAI/bge-m3"），这里视为默认模型别名
        if (model_path is None or model_path == self.model_path or model_path == "BAAI/bge-m3") and \
           (device is None or device == self.device):
            return EmbeddingPool._embedding_model
        
        # 否则创建新实例（不常见）
        print(f"[EmbeddingPool] ⚠️  使用非默认参数，创建新实例")
        return HuggingFaceEmbeddings(
            model_name=model_path or self.model_path,
            model_kwargs={"device": device or self.device, "local_files_only": True},
            encode_kwargs={"normalize_embeddings": True}
        )
    
    def get_stats(self) -> dict:
        """获取模型池统计信息"""
        return {
            'initialized': EmbeddingPool._initialized,
            'model_path': self.model_path,
            'device': self.device
        }


# 全局单例实例
_embedding_pool_instance: Optional[EmbeddingPool] = None


def get_embedding_pool() -> EmbeddingPool:
    """获取全局 Embedding 模型池实例"""
    global _embedding_pool_instance
    if _embedding_pool_instance is None:
        _embedding_pool_instance = EmbeddingPool()
    return _embedding_pool_instance


def get_pooled_embeddings(
    model_path: Optional[str] = None,
    device: Optional[str] = None
) -> HuggingFaceEmbeddings:
    """
    便捷函数：从模型池获取 Embedding 实例
    
    Examples:
        >>> embeddings = get_pooled_embeddings()  # 默认BGE-M3
        >>> embeddings = get_pooled_embeddings(device='cpu')  # CPU版本
    """
    pool = get_embedding_pool()
    return pool.get_embeddings(model_path, device)


if __name__ == "__main__":
    # 测试
    print("=== 测试 Embedding 模型池 ===\n")
    
    # 初始化模型池
    pool = get_embedding_pool()
    
    # 获取统计信息
    print("\n模型池信息:")
    print(pool.get_stats())
    
    # 测试获取实例
    print("\n测试获取实例:")
    emb1 = get_pooled_embeddings()
    emb2 = get_pooled_embeddings()
    
    print(f"两次获取是同一实例: {emb1 is emb2}")
    
    # 测试 Embedding
    print("\n测试 Embedding:")
    start = time.time()
    vector = emb1.embed_query("阿尔茨海默病")
    print(f"向量维度: {len(vector)}")
    print(f"耗时: {time.time() - start:.2f}秒")
