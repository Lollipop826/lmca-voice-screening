from __future__ import annotations

import os
import json
import shutil
from typing import List

from langchain_core.documents import Document
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.vectorstores.utils import filter_complex_metadata
    import chromadb
except ImportError:
    HuggingFaceEmbeddings = None
    Chroma = None
    filter_complex_metadata = None
    chromadb = None


def load_documents_from_jsonl_dir(directory: str) -> List[Document]:
    """从JSONL目录加载文档，每个JSONL文件包含段落级别的chunks"""
    documents: List[Document] = []
    directory = os.path.abspath(directory)
    for root, _, files in os.walk(directory):
        for name in files:
            if not name.lower().endswith(".jsonl"):
                continue
            path = os.path.join(root, name)
            with open(path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = (record.get("text") or "").strip()
                    if not text:
                        continue
                    metadata = record.get("metadata") or {}
                    if not isinstance(metadata, dict):
                        metadata = {}
                    metadata = dict(metadata)
                    metadata.setdefault("source", path)
                    metadata.setdefault("jsonl_line", line_no)
                    documents.append(Document(page_content=text, metadata=metadata))
    return documents


def build_paragraph_vector_db(
    chunks_dir: str,
    persist_dir: str,
    collection_name: str = "ad_kb",
    embedding_model: str = "BAAI/bge-m3",
    device: str = "cuda",
    wipe: bool = True,
) -> int:
    """构建段落级向量数据库（双编码器）"""
    docs = load_documents_from_jsonl_dir(chunks_dir)
    
    if wipe and os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    # 使用池化的 Embedding 模型（避免重复加载）
    from .embedding_pool import get_pooled_embeddings
    embeddings = get_pooled_embeddings(model_path=embedding_model, device=device)
    
    # 创建持久化客户端
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    
    vectordb = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    
    if docs:
        # 过滤复杂的metadata（列表、字典等）
        filtered_docs = filter_complex_metadata(docs)
        vectordb.add_documents(filtered_docs)
        # 新版Chroma自动持久化，不需要手动调用persist()
    
    return len(docs)


def get_paragraph_retriever(
    persist_dir: str,
    collection_name: str = "ad_kb",
    embedding_model: str = "BAAI/bge-m3",
    device: str = "cuda",
    k: int = 20,
):
    """获取段落级检索器（双编码器）"""
    # 使用池化的 Embedding 模型（避免重复加载）
    from .embedding_pool import get_pooled_embeddings
    embeddings = get_pooled_embeddings(model_path=embedding_model, device=device)
    
    # 使用持久化客户端
    client = chromadb.PersistentClient(path=persist_dir)
    
    vectordb = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    
    return vectordb.as_retriever(search_kwargs={"k": k})


def paragraph_retrieval(
    query: str,
    persist_dir: str,
    collection_name: str = "ad_kb",
    embedding_model: str = "BAAI/bge-m3",
    device: str = "cuda",
    k: int = 20,
) -> List[Document]:
    """段落级检索（双编码器）- 根据查询返回最相关的段落"""
    retriever = get_paragraph_retriever(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        device=device,
        k=k,
    )
    
    # 新版 LangChain 使用 invoke() 方法
    return retriever.invoke(query)
