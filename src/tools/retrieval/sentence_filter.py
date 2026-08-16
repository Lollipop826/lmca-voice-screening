from __future__ import annotations

import re
from typing import List

try:
    import torch
    from sentence_transformers import CrossEncoder
except ImportError:
    torch = None
    CrossEncoder = None


DEFAULT_MODEL = "BAAI/bge-reranker-base"


def split_sentences(text: str) -> List[str]:
    # Simple zh/en sentence split
    # Split by Chinese/English sentence endings, with or without following whitespace
    s = re.split(r"(?<=[。！？!?.])\s*|\n+", text.strip())
    return [t.strip() for t in s if t and t.strip()]


class SentenceFilter:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        # 强制使用本地文件（禁止联网下载）
        try:
            self.model = CrossEncoder(model_name, device=(device or "cpu"), local_files_only=True)
            print(f"[SentenceFilter] ✅ 成功加载本地模型: {model_name}")
        except Exception as e:
            print(f"[SentenceFilter] ⚠️ 本地模型加载失败，弹性降级: {type(e).__name__}")
            # 创建简单假模型（不做过滤）
            self.model = None

    def keep(self, query: str, text: str, threshold: float = 0.3) -> str:
        """Return concatenated sentences whose score >= threshold."""
        sentences = split_sentences(text)
        if not sentences:
            return ""
            
        # 如果模型为None（加载失败），直接返回原文本（不过滤）
        if self.model is None:
            return text
            
        # 正常过滤流程
        pairs = [[query, s] for s in sentences]
        with torch.inference_mode():
            scores = self.model.predict(pairs).tolist()
        kept = [s for s, sc in zip(sentences, scores) if sc >= threshold]
        return "\n".join(kept)

