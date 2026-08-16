"""
查询生成工具 - 供Agent调用

根据当前维度、对话历史、用户画像等信息，生成用于知识检索的查询语句
"""

from __future__ import annotations

from typing import List, Optional, Type, Union
import json

from pydantic import BaseModel, Field, PrivateAttr, field_validator
from langchain_core.tools import BaseTool

from src.common.types import InfoDimension, ConversationTurn, Profile, SearchQueryResult
from src.tools.query_sentence.generator import QuerySentenceGenerator, QuerySentenceGeneratorConfig


class QueryToolArgs(BaseModel):
    """查询生成工具参数"""
    
    dimension: Union[dict, str] = Field(
        ..., 
        description="当前评估的维度，必须包含name字段，如{'name': '定向力', 'description': '时间/地点定向'}"
    )
    history: Optional[List[dict]] = Field(
        default=None, 
        description="对话历史列表，格式如[{'role': 'user', 'content': '...'}]"
    )
    last_emotion: Optional[str] = Field(
        default=None, 
        description="用户上次情绪，如'worried'、'calm'、'anxious'等"
    )
    profile: Optional[dict] = Field(
        default=None, 
        description="用户画像，如{'age': 70, 'education_years': 6}"
    )
    
    @field_validator('dimension', mode='before')
    @classmethod
    def parse_dimension(cls, v):
        """自动解析字符串为字典"""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except:
                return {"name": "未知维度", "id": "unknown"}
        return v


class QueryTool(BaseTool):
    """
    查询生成工具
    
    根据当前维度、对话历史、用户情绪和画像，生成用于知识检索的查询语句。
    """
    
    name: str = "query_sentence_generator"
    description: str = (
        "根据给定维度、历史对话、用户情绪和画像，生成用于知识检索的查询语句。"
        "输入维度信息和对话上下文，输出优化的检索查询（关键词组合形式）。"
        "适用场景：在检索知识前，先生成合适的查询语句。"
    )
    
    args_schema: Type[BaseModel] = QueryToolArgs
    
    _generator: QuerySentenceGenerator = PrivateAttr()
    
    def __init__(self, config: Optional[QuerySentenceGeneratorConfig] = None, use_local: bool = False, llm_instance = None, **kwargs):
        super().__init__(**kwargs)
        
        llm = None
        if use_local and llm_instance:
            llm = llm_instance
        elif use_local:
            from src.llm.model_pool import get_pooled_llm
            # 🔥 改用7B模型（precise模式：temperature=0.3, max_tokens=256）替代0.5B小模型
            # 原因：0.5B模型在某些情况下会生成重复token导致解码为空
            llm = get_pooled_llm(pool_key='precise')  # 使用 7B-GPTQ 模型，更稳定可靠
            
        self._generator = QuerySentenceGenerator(config=config, llm=llm)
    
    def _run(
        self,
        dimension: dict,
        history: List[dict] = None,
        last_emotion: Optional[str] = None,
        profile: Optional[dict] = None
    ) -> str:  # type: ignore[override]
        """
        执行查询生成
        
        Returns:
            JSON格式的查询结果
        """
        import time
        _start_time = time.time()
        print(f"\n⏱️  [QueryTool] 开始生成检索查询")
        # 容错处理：如果dimension是字符串，尝试解析成字典
        if isinstance(dimension, str):
            try:
                dimension = json.loads(dimension)
            except:
                # 如果解析失败，返回错误
                return json.dumps({
                    "query": "阿尔茨海默病 认知评估",
                    "keywords": ["阿尔茨海默病", "认知评估"],
                    "error": "dimension参数解析失败，使用默认查询"
                }, ensure_ascii=False)
        
        # 转换参数类型
        dim: InfoDimension = dimension
        hist: List[ConversationTurn] = history or []
        prof: Optional[Profile] = profile
        
        # 生成查询
        result: SearchQueryResult = self._generator.generate_query(
            dimension=dim,
            history=hist,
            last_emotion=last_emotion,
            profile=prof
        )
        
        # 返回JSON格式
        _elapsed = time.time() - _start_time
        print(f"✅ [QueryTool] 查询生成完成 (耗时: {_elapsed:.2f}秒)\n")
        return json.dumps(result, ensure_ascii=False)
    
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")

