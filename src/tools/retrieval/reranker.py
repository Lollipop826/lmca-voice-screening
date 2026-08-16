from __future__ import annotations

from typing import List, Tuple, Dict, Any

try:
    import torch
    from sentence_transformers import CrossEncoder
except ImportError:
    torch = None
    CrossEncoder = None


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


class CrossEncoderReranker:
    """Query-passage cross-encoder reranker.

    Uses a lightweight Chinese/English-capable BGE reranker by default.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, device: str | None = None):
        if device is None:
            device = "cpu"
        self.device = device
        # 强制使用本地文件（禁止联网下载）
        try:
            self.model = CrossEncoder(model_name, device=self.device, local_files_only=True)
            print(f"[Reranker] ✅ 成功加载本地模型: {model_name}")
        except Exception as e:
            print(f"[Reranker] ⚠️ 本地模型加载失败，禁用精排功能: {type(e).__name__}")
            self.model = None

    def score(self, query: str, passages: List[str], batch_size: int = 16) -> List[float]:
        # 如果模型为None（加载失败），返回全部为1的假分数（保留原排序）
        if self.model is None:
            return [1.0] * len(passages)
            
        # 正常打分流程
        pairs = [[query, p] for p in passages]
        with torch.inference_mode():
            scores = self.model.predict(pairs, batch_size=batch_size).tolist()
        return scores

    def rerank(
        self,
        query: str,
        items: List[Tuple[str, Dict[str, Any]]],
        top_k: int = 20,
        batch_size: int = 16,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """Rerank (text, metadata) items by cross-encoder score.

        Returns list sorted by score desc with appended score.
        """
        # 如果模型为None（加载失败），则维持原来的排序
        if self.model is None:
            # 全部返回加上相同的分数（保持原有排序）
            return [(t, m, 1.0) for t, m in items[:top_k]]
            
        # 正常精排流程
        texts = [t for t, _ in items]
        scores = self.score(query, texts, batch_size=batch_size)
        triplets = [(t, m, s) for (t, m), s in zip(items, scores)]
        triplets.sort(key=lambda x: x[2], reverse=True)
        return triplets[:top_k]

