"""
知识检索工具 - 供Agent调用
支持RAG Fusion多查询融合检索
"""

from typing import Type, Optional
import json
import os

from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from src.llm.http_client_pool import get_siliconflow_chat_openai, get_volcengine_chat_openai

from src.tools.retrieval.paragraph_retrieval import paragraph_retrieval
from src.tools.retrieval.sentence_filter import SentenceFilter, split_sentences
from src.tools.retrieval.rag_fusion import create_rag_fusion_retriever, RAGFusion
from src.tools.retrieval.reranker import CrossEncoderReranker
from src.utils.tool_logger import ToolLogger
from .retrieval_cache import RetrievalCache
try:
    import torch
except ImportError:
    torch = None


class KnowledgeRetrievalToolArgs(BaseModel):
    """知识检索工具参数"""
    query: str = Field(..., description="检索查询语句，应该是关键词组合，如'阿尔茨海默病 定向力 评估'")
    top_k: int = Field(default=5, description="返回多少个相关文档，默认5个")
    use_fusion: bool = Field(default=True, description="是否使用RAG Fusion技术（多查询融合）")


class KnowledgeRetrievalTool(BaseTool):
    """
    知识检索工具
    
    从医学知识库中检索与查询相关的专业知识。
    支持两种模式：
    1. 标准检索：单查询向量检索
    2. RAG Fusion：多查询融合检索（提升召回率15-30%）
    """
    
    name: str = "knowledge_retrieval"
    description: str = (
        "从阿尔茨海默病医学知识库中检索相关专业知识。"
        "输入检索查询（关键词组合），返回最相关的文档段落。"
        "适用场景：需要专业医学知识来支持问题生成或回答评估时使用。"
    )
    
    args_schema: Type[BaseModel] = KnowledgeRetrievalToolArgs
    
    vector_db_dir: str = "kb/.chroma_semantic"
    collection_name: str = "ad_kb_semantic"
    # Use a stable alias handled by EmbeddingPool (maps to the local BGE-M3 path).
    embedding_model: str = "BAAI/bge-m3"
    
    # RAG Fusion配置（已关闭，生成变体耗时4+秒）
    enable_rag_fusion: bool = False  # 关闭RAG Fusion，使用标准检索更快
    fusion_num_queries: int = 5  # 生成查询数量
    fusion_docs_per_query: int = 8  # 每个查询召回数
    fusion_enable_reranking: bool = False  # 是否在fusion后精排
    
    _sentence_filter: SentenceFilter = PrivateAttr(default_factory=SentenceFilter)
    _cache: RetrievalCache = PrivateAttr(default_factory=lambda: RetrievalCache(maxsize=50, ttl=1800))
    _rag_fusion: Optional[RAGFusion] = PrivateAttr(default=None)
    _llm: Optional[ChatOpenAI] = PrivateAttr(default=None)
    _reranker: Optional[CrossEncoderReranker] = PrivateAttr(default=None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sentence_filter = SentenceFilter()
        self._cache = RetrievalCache(maxsize=50, ttl=1800)  # 30分钟缓存
        
        # 初始化RAG Fusion（如果启用）
        if self.enable_rag_fusion:
            self._init_rag_fusion()
    
    def _init_rag_fusion(self):
        """初始化RAG Fusion组件"""
        try:
            # 初始化LLM
            if os.getenv("ARK_API_KEY"):
                self._llm = get_volcengine_chat_openai(
                    model=os.getenv("RETRIEVAL_MODEL", "doubao-seed-2-0-mini-260215"),
                    temperature=0.3,
                    timeout=15,
                    max_retries=1,
                )
            else:
                self._llm = get_siliconflow_chat_openai(
                    model=os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
                    temperature=0.3,
                    timeout=15,
                    max_retries=1,
                )
            
            # 初始化精排器（如果需要）
            if self.fusion_enable_reranking:
                self._reranker = CrossEncoderReranker()
            
            # 创建基础检索函数
            def base_retriever(query: str, k: int):
                return paragraph_retrieval(
                    query=query,
                    persist_dir=self.vector_db_dir,
                    collection_name=self.collection_name,
                    embedding_model=self.embedding_model,
                    k=k
                )
            
            # 创建RAG Fusion实例
            self._rag_fusion = create_rag_fusion_retriever(
                llm=self._llm,
                base_retriever_func=base_retriever,
                num_queries=self.fusion_num_queries,
                docs_per_query=self.fusion_docs_per_query,
                enable_reranking=self.fusion_enable_reranking,
                reranker=self._reranker,
                final_top_k=5,
                verbose=True,
            )
            
            print("[RETRIEVAL] ✅ RAG Fusion已启用")
            
        except Exception as e:
            print(f"[RETRIEVAL] ⚠️ RAG Fusion初始化失败: {type(e).__name__}，将使用标准检索")
            self._rag_fusion = None
    
    def _run(self, query: str, top_k: int = 5, skip_reranking: bool = True, use_fusion: bool = True) -> str:
        """
        执行检索
        
        Args:
            query: 检索查询
            top_k: 返回结果数量
            skip_reranking: 是否跳过重排序（True=跳过、更快）
            use_fusion: 是否使用RAG Fusion
        
        Returns:
            JSON格式的检索结果
        """
        # 🔥 短路：当 USE_LOCAL_EMBEDDING=false 时（API 模式或无本地 embedding），
        # 不要尝试加载 BGE-M3 / 调用 chroma，直接返回空结果，避免每轮 CUDA OOM 浪费 ~1s
        if os.getenv("USE_LOCAL_EMBEDDING", "true").lower() != "true":
            return json.dumps({
                "success": True,
                "query": query,
                "results_count": 0,
                "results": [],
                "skipped": "USE_LOCAL_EMBEDDING=false",
            }, ensure_ascii=False)

        retrieval_mode = 'RAG Fusion' if (use_fusion and self._rag_fusion and self.enable_rag_fusion) else '标准检索'
        logger = ToolLogger("RetrievalTool")
        logger.start(
            query=query[:60] + ("..." if len(query) > 60 else ""),
            mode=retrieval_mode,
            top_k=top_k,
            skip_reranking=skip_reranking,
        )
        
        # 为fusion生成独特的缓存key
        cache_key = f"{query}_fusion={use_fusion}"
        cached_result = self._cache.get(cache_key, top_k)
        if cached_result:
            cached_count = "?"
            try:
                cached_payload = json.loads(cached_result)
                cached_count = cached_payload.get("results_count", "?")
            except Exception:
                pass
            logger.step("缓存命中")
            logger.end(status="cache_hit", results_count=cached_count, mode=retrieval_mode)
            return cached_result
        
        logger.step(f"检索模式: {retrieval_mode}")
        
        try:
            # 选择检索策略
            if use_fusion and self._rag_fusion and self.enable_rag_fusion:
                # RAG Fusion检索
                docs = self._rag_fusion.retrieve(
                    query=query,
                    dimension_name="",  # 可以从上下文传入
                    top_k=top_k
                )
            else:
                # 标准向量检索
                docs = paragraph_retrieval(
                    query=query,
                    persist_dir=self.vector_db_dir,
                    collection_name=self.collection_name,
                    embedding_model=self.embedding_model,
                    k=top_k
                )
        except Exception as e:
            logger.log(f"检索失败: {e}", level="error")
            # 返回空结果，不阻塞流程
            result = json.dumps({
                "success": False,
                "query": query,
                "results_count": 0,
                "results": [],
                "error": str(e)
            }, ensure_ascii=False)
            logger.end(status="error", error=str(e)[:120], results_count=0)
            return result
        
        # 2. 文档处理 - 优化：可选择跳过重排序
        results = []
        for i, doc in enumerate(docs[:top_k], 1):
            sentences = split_sentences(doc.page_content)
            if not sentences:
                continue
            
            # ⚡ 性能优化：默认跳过重排序，直接使用向量检索结果
            if skip_reranking:
                # 直接取前3句，不做句子级重排序（提速60%）
                relevant_sentences = sentences[:3]
                scores = [0.6] * len(sentences)  # 默认中等分数
            else:
                # 传统方法：对前3个文档做句子级重排序
                if i <= 3:
                    pairs = [[query, s] for s in sentences]
                    with torch.inference_mode():
                        scores = self._sentence_filter.model.predict(pairs).tolist()
                    
                    relevant_sentences = [
                        s for s, score in zip(sentences, scores) if score >= 0.5
                    ]
                    
                    if not relevant_sentences:
                        relevant_sentences = sentences[:3]
                else:
                    relevant_sentences = sentences[:3]
                    scores = [0.5] * len(sentences)
            
            results.append({
                "rank": i,
                "text": "\n".join(relevant_sentences[:3]),  # 优化：从5句减到3句
                "full_text": doc.page_content[:300],  # 优化：从500字减到300字
                "source": doc.metadata.get("filename", "未知"),
                "relevance": "high" if any(s >= 0.7 for s in scores) else "medium"
            })
        
        result = json.dumps({
            "success": True,
            "query": query,
            "results_count": len(results),
            "results": results
        }, ensure_ascii=False, indent=2)
        
        # 保存到缓存（使用fusion-aware key）
        cache_key = f"{query}_fusion={use_fusion}"
        self._cache.set(cache_key, top_k, result)
        logger.end(
            status="success",
            results_count=len(results),
            mode=retrieval_mode,
            top_source=results[0].get("source", "N/A") if results else "N/A",
        )
        
        return result
    
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")
