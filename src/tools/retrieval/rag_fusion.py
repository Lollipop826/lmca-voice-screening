"""
RAG Fusion - 多查询融合检索技术

原理：
1. 使用LLM生成多个相似查询变体（5-10个）
2. 每个查询独立检索Top K文档
3. 使用RRF（Reciprocal Rank Fusion）算法融合排序
4. 可选：精排（Reranking）
5. 返回融合后的Top K结果

参考论文：RAG Fusion (https://arxiv.org/abs/2402.03367)
"""

from typing import List, Dict, Any, Optional, Tuple
import time
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI


@dataclass
class FusionConfig:
    """RAG Fusion配置"""
    num_queries: int = 5  # 生成查询数量（5-10个）
    docs_per_query: int = 8  # 每个查询召回文档数
    rrf_k: int = 60  # RRF算法的K参数（常用60）
    enable_reranking: bool = False  # 是否启用精排
    final_top_k: int = 5  # 最终返回文档数


class RAGFusion:
    """
    RAG Fusion实现
    
    使用多查询+倒序融合排序提升召回率和准确性
    """
    
    def __init__(
        self,
        llm: ChatOpenAI,
        retriever_func: callable,
        config: Optional[FusionConfig] = None,
        reranker: Optional[Any] = None,
        verbose: bool = True,
    ):
        """
        Args:
            llm: 用于生成查询变体的大模型
            retriever_func: 检索函数，签名为 func(query: str, k: int) -> List[Document]
            config: Fusion配置
            reranker: 精排器（可选），需要有score方法
            verbose: 是否打印详细日志
        """
        self.llm = llm
        self.retriever_func = retriever_func
        self.config = config or FusionConfig()
        self.reranker = reranker
        self.verbose = verbose
    
    def generate_queries(self, original_query: str, dimension_name: str = "") -> List[str]:
        """
        生成多个查询变体
        
        Args:
            original_query: 原始查询
            dimension_name: 维度名称（用于上下文）
        
        Returns:
            包含原始查询在内的查询列表
        """
        start_time = time.time()
        
        prompt = f"""你是医学检索查询优化专家。我需要你生成{self.config.num_queries - 1}个与原始查询相似但角度不同的查询变体。

【原始查询】
{original_query}

【评估维度】
{dimension_name if dimension_name else "阿尔茨海默病认知评估"}

【生成要求】
1. 生成{self.config.num_queries - 1}个查询变体（不包括原查询）
2. 每个变体应该：
   - 保持医学专业性
   - 从不同角度表达相同意图
   - 使用同义词或相关术语
   - 可以是更具体或更抽象的表达
3. 变体示例：
   - 原查询："阿尔茨海默病 记忆力 评估"
   - 变体1："老年痴呆 记忆障碍 诊断标准"
   - 变体2："AD 短期记忆 认知筛查"
   - 变体3："失智症 遗忘 临床评估"

【输出格式】
每行一个查询，不要编号，不要解释，只输出查询文本。
"""
        
        try:
            if self.verbose:
                print(f"[RAG_FUSION] 🔄 调用LLM生成查询变体...")
            
            response = self.llm.invoke(prompt)
            
            # 详细检查响应对象
            if self.verbose:
                print(f"[RAG_FUSION] 📥 LLM响应类型: {type(response)}")
                print("[RAG_FUSION] 📥 LLM响应对象已收到")
                if hasattr(response, 'content'):
                    print(f"[RAG_FUSION] 📥 response.content类型: {type(response.content)}")
                    print(f"[RAG_FUSION] 📥 response.content_chars: {len(str(response.content))}")
                else:
                    print(f"[RAG_FUSION] ⚠️  响应对象没有content属性")
                    print("[RAG_FUSION] 📥 响应对象属性不可用")
            
            # 检查响应内容是否为空
            if not response:
                if self.verbose:
                    print(f"[RAG_FUSION] ⚠️  LLM返回None响应对象，使用原查询")
                return [original_query]
            
            if not hasattr(response, 'content'):
                if self.verbose:
                    print(f"[RAG_FUSION] ⚠️  LLM响应对象没有content属性，使用原查询")
                return [original_query]
            
            if response.content is None:
                if self.verbose:
                    print(f"[RAG_FUSION] ⚠️  LLM返回content=None，使用原查询")
                return [original_query]
            
            content = response.content.strip()
            
            # 如果内容为空，直接返回原查询
            if not content:
                if self.verbose:
                    print("[RAG_FUSION] ⚠️ LLM返回空字符串，使用原查询")
                return [original_query]
            
            # 解析生成的查询
            generated = [line.strip() for line in content.split('\n') if line.strip()]
            
            # 清理：去掉可能的编号
            cleaned = []
            for q in generated:
                # 去掉 "1. ", "- ", "• " 等前缀
                q = q.lstrip('0123456789.-•*> ')
                q = q.strip('"\'')
                if q and len(q) > 5:  # 至少5个字符
                    cleaned.append(q)
            
            # 组合：原查询 + 生成的查询
            # 确保至少返回原查询（即使cleaned为空）
            all_queries = [original_query] + cleaned[:self.config.num_queries - 1]
            
            # 再次检查：确保返回的查询列表不为空
            if not all_queries:
                if self.verbose:
                    print(f"[RAG_FUSION] ⚠️  解析后查询列表为空，使用原查询")
                all_queries = [original_query]
            
            if self.verbose:
                elapsed = time.time() - start_time
                print(f"[RAG_FUSION] 🔄 生成{len(all_queries)}个查询变体 ({elapsed:.2f}秒)")
                print(f"[RAG_FUSION] 查询长度总和: {sum(len(q) for q in all_queries)}")
            
            return all_queries
            
        except Exception as e:
            # 详细记录异常信息，帮助诊断问题
            error_type = type(e).__name__
            
            if self.verbose:
                print(f"[RAG_FUSION] ❌ 查询生成异常:")
                print(f"  - 异常类型: {error_type}")
                print("  - 异常详情已脱敏")
                
                # 检查是否是超时错误
                if "Timeout" in error_type:
                    print(f"  - 🔍 原因: API调用超时（timeout=15秒），可能是网络慢或服务响应慢")
                elif "Connection" in error_type:
                    print(f"  - 🔍 原因: 网络连接问题")
                else:
                    print("  - 🔍 原因: 其他异常")
                
                print(f"[RAG_FUSION] ⚠️  使用原查询作为fallback")
            
            # Fallback: 确保至少返回原查询
            return [original_query] if original_query else []
    
    def reciprocal_rank_fusion(
        self,
        query_docs_map: Dict[str, List[Document]],
        k: int = 60
    ) -> List[Tuple[Document, float]]:
        """
        倒序融合排序（RRF）
        
        公式：RRF(d) = Σ(1 / (k + rank(d, q))) for all queries q
        
        Args:
            query_docs_map: {query: [docs]} 映射
            k: RRF常数，通常设为60
        
        Returns:
            [(doc, score), ...] 排序后的文档列表
        """
        start_time = time.time()
        
        # 文档唯一标识 -> Document对象
        doc_map: Dict[str, Document] = {}
        # 文档唯一标识 -> RRF分数
        rrf_scores: Dict[str, float] = defaultdict(float)
        
        for query, docs in query_docs_map.items():
            for rank, doc in enumerate(docs, start=1):
                # 使用文档内容作为唯一标识（如果有ID更好）
                doc_id = doc.metadata.get('id') or doc.page_content[:100]
                
                # RRF公式：1 / (k + rank)
                score = 1.0 / (k + rank)
                rrf_scores[doc_id] += score
                
                # 保存文档对象
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc
        
        # 按分数排序
        sorted_items = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 构造结果
        results = [(doc_map[doc_id], score) for doc_id, score in sorted_items]
        
        if self.verbose:
            elapsed = time.time() - start_time
            print(f"[RAG_FUSION] 🔀 RRF融合完成: {len(results)}个唯一文档 ({elapsed:.2f}秒)")
        
        return results
    
    def rerank_documents(
        self,
        query: str,
        doc_score_pairs: List[Tuple[Document, float]],
        top_k: int
    ) -> List[Tuple[Document, float]]:
        """
        精排（可选）
        
        Args:
            query: 原始查询
            doc_score_pairs: [(doc, rrf_score), ...]
            top_k: 返回数量
        
        Returns:
            精排后的文档列表
        """
        if not self.reranker:
            return doc_score_pairs[:top_k]
        
        start_time = time.time()
        
        try:
            # 准备精排
            docs = [doc for doc, _ in doc_score_pairs]
            texts = [doc.page_content for doc in docs]
            
            # 调用精排器
            rerank_scores = self.reranker.score(query, texts)
            
            # 组合RRF分数和精排分数（加权）
            # 70% 精排分数 + 30% RRF分数
            final_pairs = []
            for (doc, rrf_score), rerank_score in zip(doc_score_pairs, rerank_scores):
                combined_score = 0.7 * rerank_score + 0.3 * rrf_score
                final_pairs.append((doc, combined_score))
            
            # 重新排序
            final_pairs.sort(key=lambda x: x[1], reverse=True)
            
            if self.verbose:
                elapsed = time.time() - start_time
                print(f"[RAG_FUSION] 🎯 精排完成 ({elapsed:.2f}秒)")
            
            return final_pairs[:top_k]
            
        except Exception as e:
            if self.verbose:
                print(f"[RAG_FUSION] ⚠️ 精排失败: {type(e).__name__}，使用RRF结果")
            return doc_score_pairs[:top_k]
    
    def retrieve(
        self,
        query: str,
        dimension_name: str = "",
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        执行RAG Fusion检索
        
        Args:
            query: 原始查询
            dimension_name: 维度名称
            top_k: 最终返回文档数（默认使用config）
        
        Returns:
            融合后的Top K文档列表
        """
        total_start = time.time()
        top_k = top_k or self.config.final_top_k
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[RAG_FUSION] 🚀 开始RAG Fusion检索")
            print(f"[RAG_FUSION] 📝 原始查询: text_chars={len(query)}")
            print(f"{'='*60}")
        
        # 步骤1: 生成查询变体
        queries = self.generate_queries(query, dimension_name)
        
        # 检查查询列表是否为空
        if not queries:
            if self.verbose:
                print(f"[RAG_FUSION] ⚠️  查询列表为空，无法执行检索")
            return []
        
        # 确保至少有一个有效查询
        queries = [q for q in queries if q and q.strip()]
        if not queries:
            if self.verbose:
                print(f"[RAG_FUSION] ⚠️  所有查询都为空，使用原查询")
            queries = [query] if query and query.strip() else []
            if not queries:
                return []
        
        # 步骤2: 多查询检索（并行优化）
        query_docs_map = {}
        retrieval_start = time.time()
        
        # 使用线程池并行执行多个查询的检索
        def retrieve_single_query(query_idx: int, query: str) -> Tuple[int, str, List[Document]]:
            """单个查询的检索函数"""
            try:
                docs = self.retriever_func(query, self.config.docs_per_query)
                return (query_idx, query, docs)
            except Exception as e:
                if self.verbose:
                    print(f"[RAG_FUSION] ⚠️ 查询{query_idx}检索失败: {type(e).__name__}")
                return (query_idx, query, [])
        
        # 并行执行所有查询（最多5个线程）
        with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as executor:
            future_to_query = {
                executor.submit(retrieve_single_query, i, q): (i, q) 
                for i, q in enumerate(queries, 1)
            }
            
            # 按完成顺序收集结果（保持顺序）
            results = [None] * len(queries)
            for future in as_completed(future_to_query):
                query_idx, query, docs = future.result()
                results[query_idx - 1] = (query_idx, query, docs)
        
        # 填充结果映射
        for query_idx, query, docs in results:
            query_docs_map[query] = docs
            if self.verbose:
                print(f"[RAG_FUSION] 📥 查询{query_idx}检索到 {len(docs)} 个文档")
        
        retrieval_time = time.time() - retrieval_start
        if self.verbose:
            total_docs = sum(len(docs) for docs in query_docs_map.values())
            print(f"[RAG_FUSION] ✅ 总共检索 {total_docs} 个文档 ({retrieval_time:.2f}秒)")
        
        # 步骤3: RRF融合排序
        fused_docs = self.reciprocal_rank_fusion(
            query_docs_map,
            k=self.config.rrf_k
        )
        
        # 步骤4: 可选精排
        if self.config.enable_reranking and self.reranker:
            final_docs = self.rerank_documents(query, fused_docs, top_k)
        else:
            final_docs = fused_docs[:top_k]
        
        # 只返回Document对象
        results = [doc for doc, _ in final_docs]
        
        total_time = time.time() - total_start
        if self.verbose:
            print(f"[RAG_FUSION] 🎉 完成！返回 {len(results)} 个文档 (总耗时: {total_time:.2f}秒)")
            print(f"{'='*60}\n")
        
        return results


def create_rag_fusion_retriever(
    llm: ChatOpenAI,
    base_retriever_func: callable,
    num_queries: int = 5,
    docs_per_query: int = 8,
    enable_reranking: bool = False,
    reranker: Optional[Any] = None,
    final_top_k: int = 5,
    verbose: bool = True,
) -> RAGFusion:
    """
    便捷工厂方法：创建RAG Fusion检索器
    
    Args:
        llm: 大模型
        base_retriever_func: 基础检索函数
        num_queries: 生成查询数量
        docs_per_query: 每个查询召回数
        enable_reranking: 是否精排
        reranker: 精排器
        final_top_k: 最终返回数量
        verbose: 详细日志
    
    Returns:
        RAGFusion实例
    """
    config = FusionConfig(
        num_queries=num_queries,
        docs_per_query=docs_per_query,
        enable_reranking=enable_reranking,
        final_top_k=final_top_k,
        rrf_k=60,
    )
    
    return RAGFusion(
        llm=llm,
        retriever_func=base_retriever_func,
        config=config,
        reranker=reranker,
        verbose=verbose,
    )
