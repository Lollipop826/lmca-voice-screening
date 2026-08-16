from __future__ import annotations

from typing import List, Optional
import os
import logging
import re

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from src.common.types import InfoDimension, ConversationTurn, SearchQueryResult, Profile
from src.llm.http_client_pool import get_siliconflow_chat_openai


# Provider defaults
DEFAULT_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
DEFAULT_API_KEY = os.getenv("SILICONFLOW_API_KEY")
DEFAULT_MODEL = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-14B-Instruct")


class QuerySentenceGeneratorConfig(BaseModel):
    model: str = Field(default=DEFAULT_MODEL, description="LLM model name")
    temperature: float = 0.2
    max_query_chars: int = 40
    base_url: Optional[str] = Field(default=os.getenv("OHMYGPT_BASE_URL", os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)))
    api_key: Optional[str] = Field(default=DEFAULT_API_KEY)
    # Optimized System Prompt for keyword-combination style
    system_prompt: str = (
        "你是一个专业的阿尔茨海默病（AD）初筛检索助手。你的任务是基于用户画像、历史对话、上次情绪和指定维度，"
        "生成一条简洁的中文检索查询语句。该语句专用于后续RAG知识检索，帮助查找与该维度相关的评估或补充信息。\n"
        "规则：\n"
        "- 查询语句应是关键词组合形式，如‘阿尔茨海默病 [维度] [用户线索] [画像相关] 评估’，适合搜索引擎或向量检索。\n"
        "- 结合用户画像（如年龄、教育水平）添加相关关键词（如对老人加‘老年’）。\n"
        "- 从历史对话中提取关键线索，避免重复；融入查询以精准化（如历史提到‘忘记东西’，加‘记忆遗忘’）。\n"
        "- 考虑上次情绪作为检索焦点（如焦虑，可加‘情绪影响’或‘心理症状’）。\n"
        "- 焦点围绕指定维度（如维度为‘记忆力’，核心词围绕‘记忆评估’）。\n"
        "- 输出只是一条检索查询语句，不要像问诊问题一样加‘你/您’等称呼，不要添加解释、标点装饰或额外文本。\n"
        "- 保持10-20字以内，用空格分隔关键词，简明高效。"
    )
    verbose: bool = True
    log_level: str = Field(default=os.getenv("QS_LOG_LEVEL", "INFO"))


class QuerySentenceGenerator:
    """Generate an internal query sentence (non user-facing) for next step planning."""

    def __init__(self, config: Optional[QuerySentenceGeneratorConfig] = None, llm: Optional[ChatOpenAI] = None):
        self.config = config or QuerySentenceGeneratorConfig()

        # Logger setup
        self.logger = logging.getLogger("tools.query_sentence")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)
        level_name = self.config.log_level.upper() if self.config.log_level else ("DEBUG" if self.config.verbose else "INFO")
        self.logger.setLevel(getattr(logging, level_name, logging.INFO))

        if llm is not None:
            self.llm = llm
        else:
            self.logger.info(f"Init LLM model={self.config.model} base_url={self.config.base_url}")
            self.llm = get_siliconflow_chat_openai(
                model=self.config.model,
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                temperature=self.config.temperature,
                timeout=20,
                max_retries=1,
            )

    def _fallback(self, dim: Optional[InfoDimension]) -> SearchQueryResult:
        self.logger.warning("LLM unavailable or invalid output; using fallback query.")
        if not dim:
            return SearchQueryResult(
                query="阿尔茨海默病 关键信息 评估",
                keywords=["阿尔茨海默病", "关键信息", "评估"],
                target_dimensions=[],
                confidence=0.3,
                rationale="rule_default",
                used_fallback=True,
            )
        name = dim.get("name") or dim.get("id") or "症状"
        q = f"阿尔茨海默病 {name} 评估"
        return SearchQueryResult(
                query=q,
                keywords=q.split(),
                target_dimensions=[dim.get("id", name)],
                confidence=0.5,
                rationale="rule_dimension",
                used_fallback=True,
        )


    def _sanitize_query(self, text: str) -> str:
        t = text.strip().strip('"').strip("'")
        # Remove pronouns 你/您
        t = re.sub(r"[你您]", "", t)
        # Replace punctuation with space and normalize spaces
        t = re.sub(r"[，。；、,.!！?？：:]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        # Enforce max length (approx chars)
        if len(t) > self.config.max_query_chars:
            t = t[: self.config.max_query_chars].strip()
        return t

    def generate_query(
        self,
        dimension: InfoDimension,
        history: Optional[List[ConversationTurn]] = None,
        last_emotion: Optional[str] = None,
        profile: Optional[Profile] = None,
    ) -> SearchQueryResult:
        # Log inputs (compact)
        self.logger.info(
            "Input dimension id=%s status=%s priority=%s",
            dimension.get("id"),
            dimension.get("status"),
            dimension.get("priority"),
        )
        self.logger.info(f"History turns={len(history or [])} last_emotion={last_emotion}")
        if profile:
            self.logger.info(
                "Profile fields age=%s edu=%s notes_chars=%s",
                profile.get("age"),
                profile.get("education_years"),
                len(str(profile.get("notes") or "")),
            )

        # Prepare template vars
        dim_info = f"id: {dimension.get('id')}, name: {dimension.get('name')}, status: {dimension.get('status')}, priority: {dimension.get('priority')}, value: {dimension.get('value')}"
        profile_hint = ""
        if profile:
            profile_hint = f"年龄:{profile.get('age')}, 教育年限:{profile.get('education_years')}, 备注:{profile.get('notes')}"
        history_compact = [
            {"role": t.get("role"), "content": t.get("content"), "emotion": t.get("emotion")}
            for t in (history or [])
        ]

        # Build user prompt per optimized template
        user_prompt = (
            "当前维度信息: {dimension_info}\n"
            "用户画像: {profile_hint}\n"
            "历史对话(截断): {history_compact}\n"
            "上次用户情绪: {last_emotion}\n\n"
            "基于以上，生成一条面向RAG检索的查询语句，焦点是该维度的关键词组合。"
        ).format(
            dimension_info=dim_info,
            profile_hint=profile_hint,
            history_compact=history_compact,
            last_emotion=last_emotion,
        )

        self.logger.debug("Prompt chars=%s", len(user_prompt))

        try:
            msg = self.llm.invoke([
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": user_prompt},
            ])
            content = getattr(msg, "content", "") or ""
            query = self._sanitize_query(content)
            self.logger.info("LLM output chars=%s", len(content))
            if not query:
                self.logger.warning("LLM output empty; switching to fallback.")
                return self._fallback(dimension)
            result = SearchQueryResult(
                query=query,
                keywords=query.split(),
                target_dimensions=[dimension.get("id")] if dimension.get("id") else [],
                confidence=0.7,
                rationale="llm",
                used_fallback=False,
            )
            self.logger.info(
                "Result query_chars=%s keywords=%s",
                len(query),
                len(result["keywords"]),
            )
            return result
        except Exception as e:
            self.logger.error("LLM invocation failed: %s", type(e).__name__)
            return self._fallback(dimension)
