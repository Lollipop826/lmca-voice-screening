"""
问题生成工具 - 供Agent调用
"""

import threading
from typing import Type, Optional, List, Dict
import json
import os
from datetime import datetime

from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from src.utils.location_service import get_realtime_context
from src.llm.http_client_pool import (
    create_volcengine_context_cache,
    get_chat_openai,
    get_volcengine_context_chat_openai,
)


class QuestionGenerationToolArgs(BaseModel):
    """问题生成工具参数"""
    dimension_name: str = Field(..., description="当前评估的维度名称，如'定向力'、'记忆力'等")
    dimension_description: str = Field(default="", description="维度的描述，如'时间/地点定向'")
    knowledge_context: str = Field(
        default="", 
        description="检索到的相关医学知识，用于指导问题生成"
    )
    patient_age: Optional[int] = Field(default=None, description="患者年龄")
    patient_education: Optional[int] = Field(default=None, description="患者教育年限")
    patient_name: Optional[str] = Field(default=None, description="患者姓名，用于称呼患者")
    patient_gender: Optional[str] = Field(default=None, description="患者性别：男/女")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None, 
        description="最近的对话历史（JSON格式列表）"
    )
    generated_questions: Optional[List[str]] = Field(
        default=None, 
        description="已生成的问题列表（用于去重）"
    )
    patient_emotion: Optional[str] = Field(default=None, description="患者当前情绪")
    task_instruction: Optional[str] = Field(default=None, description="本轮任务指令（内部用，不向对方暴露）")
    task_id: Optional[str] = Field(default=None, description="当前任务ID，用于固定目标问题兜底归一化")
    persona_hooks: Optional[List[str]] = Field(default=None, description="个性化钩子（兴趣/习惯/刚聊到的点）")
    must_include: Optional[List[str]] = Field(default=None, description="生成问题必须包含的关键词/数字")
    avoid_questions: Optional[List[str]] = Field(default=None, description="需要避免的已问过问题（防止重复）")
    bridge_hint: Optional[str] = Field(default=None, description="自然过渡提示，用于约束问题目标")
    target_question: Optional[str] = Field(default=None, description="上游已确定的固定目标问题，若提供则工具只负责生成回应并自然衔接")
    target_question_core: Optional[str] = Field(default=None, description="固定目标问题的核心问法，用于命中校验；若不提供则在工具内推导")
    conversation_summary: Optional[str] = Field(default=None, description="对话前情摘要（LLM生成+结构化状态），用于长对话记忆")


class QuestionGenerationTool(BaseTool):
    """
    问题生成工具
    
    基于当前评估维度、医学知识和患者信息，生成合适的评估问题。
    """
    
    name: str = "generate_question"
    description: str = (
        "生成用于评估患者认知功能的问题。"
        "基于当前维度、检索到的医学知识、患者画像和对话历史，生成温和、专业、易懂的问题。"
        "适用场景：需要询问患者新问题时使用。"
    )
    
    args_schema: Type[BaseModel] = QuestionGenerationToolArgs
    
    _llm: ChatOpenAI = PrivateAttr()
    _balanced_llm: Optional[ChatOpenAI] = PrivateAttr(default=None)
    _fast_llm: Optional[ChatOpenAI] = PrivateAttr(default=None)
    _default_model: str = PrivateAttr(default="Qwen/Qwen2.5-72B-Instruct")
    _balanced_model: Optional[str] = PrivateAttr(default=None)
    _fast_model: Optional[str] = PrivateAttr(default=None)
    _fast_dimensions: set = PrivateAttr(default_factory=set)
    _context_id: Optional[str] = PrivateAttr(default=None)
    _context_llm: Optional[ChatOpenAI] = PrivateAttr(default=None)
    _context_cache_tried: bool = PrivateAttr(default=False)
    _llm_temperature: float = PrivateAttr(default=0.7)
    _llm_max_tokens: int = PrivateAttr(default=160)
    _llm_timeout: float = PrivateAttr(default=20.0)
    _verbose_logs: bool = PrivateAttr(default=False)
    _logger_state: threading.local = PrivateAttr(default_factory=threading.local)
    
    def __init__(
        self,
        use_local: bool = False,  # 新增参数
        llm_instance = None,      # 允许直接传入LLM实例
        **kwargs
    ):
        super().__init__(**kwargs)
        self._verbose_logs = any(
            os.getenv(env_name, "0").strip().lower() in {"1", "true", "yes", "on"}
            for env_name in ("QUESTION_GEN_VERBOSE_LOGS", "AGENT_VERBOSE_LOGS")
        )

        if llm_instance is not None:
            self._llm = llm_instance
            self._fast_llm = None
            self._default_model = "custom_llm_instance"
            self._fast_model = None
            self._fast_dimensions = set()
            self._use_volcengine = False
            self._context_id = None
            self._context_llm = None
            self._context_cache_tried = True
            self._log_info("使用外部注入 LLM 实例")
            return

        # 正式回复优先统一走百炼；没有百炼配置时保留既有回退模型。
        dashscope_enabled = bool(os.getenv("DASHSCOPE_API_KEY"))
        self._use_volcengine = bool(os.getenv("ARK_API_KEY")) and not dashscope_enabled
        if dashscope_enabled:
            self._default_model = (
                os.getenv("QUESTION_GEN_MODEL")
                or os.getenv("DASHSCOPE_CHAT_MODEL", "qwen3.7-flash")
            )
            self._balanced_model = os.getenv("QUESTION_GEN_BALANCED_MODEL")
            self._fast_model = os.getenv("QUESTION_GEN_FAST_MODEL")
            self._log_info("使用阿里云百炼 (DashScope)")
        elif self._use_volcengine:
            self._default_model = os.getenv("QUESTION_GEN_MODEL", "doubao-seed-2-0-lite-260215")
            self._balanced_model = os.getenv("QUESTION_GEN_BALANCED_MODEL", "doubao-seed-2-0-lite-260215")
            self._fast_model = os.getenv("QUESTION_GEN_FAST_MODEL", "doubao-seed-2-0-mini-260215")
            self._log_info("使用火山引擎 (Doubao)")
        else:
            self._default_model = os.getenv("QUESTION_GEN_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
            self._balanced_model = os.getenv("QUESTION_GEN_BALANCED_MODEL", "Qwen/Qwen2.5-32B-Instruct")
            self._fast_model = os.getenv("QUESTION_GEN_FAST_MODEL", "Qwen/Qwen2.5-14B-Instruct")
            self._log_info("使用 SiliconFlow (Qwen)")

        try:
            llm_temperature = float(os.getenv("QUESTION_GEN_TEMPERATURE", "0.7"))
        except ValueError:
            llm_temperature = 0.7
        try:
            llm_max_tokens = int(os.getenv("QUESTION_GEN_MAX_TOKENS", "160"))
        except ValueError:
            llm_max_tokens = 160
        try:
            llm_timeout = float(os.getenv("QUESTION_GEN_TIMEOUT", "20"))
        except ValueError:
            llm_timeout = 20.0

        fast_dims_raw = os.getenv("QUESTION_GEN_FAST_DIMENSIONS", "")
        self._fast_dimensions = {d.strip() for d in fast_dims_raw.split(",") if d.strip()}

        self._llm = get_chat_openai(
            model=self._default_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            timeout=llm_timeout,
            streaming=True,
        )

        if self._balanced_model and self._balanced_model != self._default_model:
            self._balanced_llm = get_chat_openai(
                model=self._balanced_model,
                temperature=max(0.5, llm_temperature - 0.05),
                max_tokens=llm_max_tokens,
                timeout=llm_timeout,
                streaming=True,
            )
        else:
            self._balanced_llm = None

        if self._fast_model and self._fast_model != self._default_model:
            self._fast_llm = get_chat_openai(
                model=self._fast_model,
                temperature=max(0.3, llm_temperature - 0.1),
                max_tokens=llm_max_tokens,
                timeout=llm_timeout,
                streaming=True,
            )
        else:
            self._fast_llm = None

        fast_model_info = self._fast_model if self._fast_llm else "disabled"
        balanced_model_info = self._balanced_model if self._balanced_llm else "disabled"
        self._log_info(
            f"主模型={self._default_model} | 平衡模型={balanced_model_info} | 快速模型={fast_model_info} | 快速维度={sorted(self._fast_dimensions)}"
        )

        # 🔥 Context Cache：延迟初始化，在 _run 首次调用时用实际 system_prompt 创建缓存
        self._context_id = None
        self._context_llm = None
        self._context_cache_tried = False
        self._llm_temperature = llm_temperature
        self._llm_max_tokens = llm_max_tokens
        self._llm_timeout = llm_timeout

    def _get_active_logger(self):
        return getattr(self._logger_state, "current", None)

    def _set_active_logger(self, logger) -> None:
        self._logger_state.current = logger

    def _clear_active_logger(self) -> None:
        if hasattr(self._logger_state, "current"):
            self._logger_state.current = None

    def _emit_log(self, message: str, level: str = "info") -> None:
        logger = self._get_active_logger()
        if logger is not None:
            logger.log(message, level=level)
            return
        prefix = {
            "info": "ℹ️",
            "warn": "⚠️",
            "error": "❌",
            "success": "✅",
        }.get(level, "ℹ️")
        print(f"[QuestionGenTool] {prefix} message_chars={len(str(message))}")

    def _log_info(self, message: str, level: str = "info") -> None:
        self._emit_log(message, level=level)

    def _log_verbose(self, message: str, level: str = "info") -> None:
        if self._verbose_logs:
            self._emit_log(message, level=level)

    def _should_use_balanced_model(self, task_instruction: Optional[str], last_user_msg: Optional[str]) -> bool:
        text = f"{task_instruction or ''} {last_user_msg or ''}"
        quality_keywords = [
            "兴趣爱好", "生活习惯", "家庭", "孩子", "学校", "回忆",
            "不想回答", "不想说", "不做", "别问", "拒绝", "不愿意",
        ]
        return any(k in text for k in quality_keywords)

    def _should_avoid_fast_fixed_target_model(
        self,
        task_instruction: Optional[str],
        last_user_msg: Optional[str],
        target_question: Optional[str],
        bridge_hint: Optional[str] = None,
    ) -> bool:
        text = f"{task_instruction or ''} {last_user_msg or ''} {target_question or ''} {bridge_hint or ''}"
        quality_markers = [
            "手机", "电视", "广播", "小说", "音乐", "散步", "公园", "老朋友", "聊天",
            "兴趣", "爱好", "平时", "平常", "生活", "家里", "出门", "习惯",
            "医院", "看病", "身体", "不舒服", "难受", "检查", "挂号", "医生",
        ]
        bridge_markers = [
            "顺着", "自然", "过渡", "承接", "聊聊", "提及", "顺势", "带到",
        ]
        target_markers = ["区", "县", "城市", "地点", "哪一年", "星期", "几月", "几号", "季节"]
        normalized_last = self._normalize_for_similarity(last_user_msg or "")
        context_heavy, _ = self._diagnose_context_dependency(last_user_msg)
        has_concrete_reply = bool(normalized_last) and len(normalized_last) >= 6 and not context_heavy
        return (
            any(k in text for k in quality_markers)
            and (any(k in text for k in bridge_markers) or bool(bridge_hint))
            and any(k in text for k in target_markers)
        ) or (
            bool(bridge_hint)
            and any(k in text for k in target_markers)
            and has_concrete_reply
        )

    def _should_use_fast_structured_model(self, dimension_name: str, task_instruction: Optional[str]) -> bool:
        """
        对时间/地点这类结构化、答案空间小的问题，优先走更快模型。
        这类题不需要 32B/72B 的开放式生成能力。
        """
        normalized_dimension = (dimension_name or "").strip()
        text = f"{normalized_dimension} {task_instruction or ''}"
        structured_keywords = [
            "定向力", "星期", "周几", "几月", "几号", "日期", "季节",
            "城市", "什么区", "哪个区", "地点", "现在这个地方", "当前位置", "地址",
        ]
        return any(k in text for k in structured_keywords)

    def _is_context_dependent_reply(self, last_user_msg: Optional[str]) -> bool:
        return self._diagnose_context_dependency(last_user_msg)[0]

    def _diagnose_context_dependency(self, last_user_msg: Optional[str]) -> tuple[bool, str]:
        import re

        text = (last_user_msg or "").strip()
        if not text:
            return False, "empty_reply"

        normalized = re.sub(r"\s+", "", text)
        if not normalized:
            return False, "blank_reply"

        if len(normalized) <= 4:
            return True, f"short_reply(len={len(normalized)})"

        concrete_patterns = [
            ("digit", r"\d"),
            ("time", r"(星期|周|礼拜|年|月|号|日|春天|夏天|秋天|冬天|春季|夏季|秋季|冬季)"),
            ("location", r"(辽宁|大连|甘井子|医院|小区|公寓|楼|层|海边|家里|家中|学校)"),
        ]
        for label, pattern in concrete_patterns:
            if re.search(pattern, normalized):
                return False, f"contains_{label}"

        vague_markers = [
            "确实", "是啊", "对啊", "对呢", "可不是", "还行", "挺好", "挺开心", "开心", "高兴",
            "不错", "行吧", "可以", "知道了", "明白", "你去看看", "你去瞅一眼", "你去瞅瞅",
            "你看吧", "你猜吧", "你说吧", "你想想", "记不清", "忘了", "想不起来", "说不好",
        ]
        for marker in vague_markers:
            if marker == normalized or marker in normalized:
                return True, f"vague_marker:{marker}"

        if len(normalized) <= 10:
            return True, f"short_without_concrete(len={len(normalized)})"
        return False, f"enough_surface_info(len={len(normalized)})"

    def _select_llm_with_reason(
        self,
        dimension_name: str,
        task_instruction: Optional[str] = None,
        last_user_msg: Optional[str] = None,
        bridge_hint: Optional[str] = None,
        prefer_context_quality: bool = False,
        fixed_target_mode: bool = False,
        target_question: Optional[str] = None,
    ) -> tuple[ChatOpenAI, str, str]:
        normalized_dimension = (dimension_name or "").strip()
        if self._balanced_llm and self._should_use_balanced_model(task_instruction, last_user_msg):
            return self._balanced_llm, self._balanced_model or self._default_model, "balanced_quality_keyword"
        if prefer_context_quality:
            return self._llm, self._default_model, "default_context_heavy_ack"
        if (
            fixed_target_mode
            and self._balanced_llm
            and self._should_avoid_fast_fixed_target_model(task_instruction, last_user_msg, target_question, bridge_hint)
        ):
            return self._balanced_llm, self._balanced_model or self._default_model, "balanced_fixed_target_bridge_quality"
        if (
            self._fast_llm
            and self._use_volcengine
            and bridge_hint
            and self._should_use_fast_structured_model(dimension_name, task_instruction)
        ):
            return self._fast_llm, self._fast_model or self._default_model, "fast_structured_with_bridge_hint"
        if self._fast_llm and normalized_dimension in self._fast_dimensions:
            return self._fast_llm, self._fast_model or self._default_model, f"fast_dimension:{normalized_dimension}"
        return self._llm, self._default_model, "default_fallback"

    def _select_llm(
        self,
        dimension_name: str,
        task_instruction: Optional[str] = None,
        last_user_msg: Optional[str] = None,
        bridge_hint: Optional[str] = None,
        prefer_context_quality: bool = False,
        fixed_target_mode: bool = False,
        target_question: Optional[str] = None,
    ) -> tuple[ChatOpenAI, str]:
        selected_llm, selected_model, _ = self._select_llm_with_reason(
            dimension_name,
            task_instruction=task_instruction,
            last_user_msg=last_user_msg,
            bridge_hint=bridge_hint,
            prefer_context_quality=prefer_context_quality,
            fixed_target_mode=fixed_target_mode,
            target_question=target_question,
        )
        return selected_llm, selected_model

    def _looks_like_question(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        markers = ("吗", "呢", "么", "什么", "怎么", "哪里", "哪个", "多少", "为何", "为啥", "是否")
        return any(m in t for m in markers) or ("？" in t) or ("?" in t)

    def _keep_single_question(self, text: str) -> str:
        """压缩为单个主问句，优先保留更具体、信息量更高的那一句。"""
        import re

        q = (text or "").strip()
        if not q:
            return q
        q = q.replace("?", "？")

        # 切成若干问句片段，并记录每句前面的前缀
        matches = list(re.finditer(r'([^？]*？)', q))
        if not matches:
            return q

        question_parts = [m.group(1).strip("，。；;!！~～ ") for m in matches]
        if len(question_parts) == 1:
            chosen = question_parts[0]
            return chosen if chosen.endswith("？") else (f"{chosen}？" if self._looks_like_question(chosen) else chosen)

        def _score_question(seg: str) -> tuple[int, int]:
            text_seg = (seg or "").strip()
            info_markers = ("什么", "怎么", "哪里", "哪个", "多少", "几月", "几号", "星期", "城市", "区", "安排", "做什么", "有没有")
            weak_markers = ("是吧", "对不对", "好吗", "是不是", "还行吗", "顺心吗")
            score = 0
            if any(m in text_seg for m in info_markers):
                score += 3
            if any(m in text_seg for m in weak_markers):
                score -= 2
            if self._looks_like_question(text_seg):
                score += 1
            score += min(len(text_seg) // 8, 2)
            return score, len(text_seg)

        best_idx = max(range(len(question_parts)), key=lambda i: _score_question(question_parts[i]))
        chosen = question_parts[best_idx].rstrip("？")

        # 尝试保留真正的回应部分，丢掉前面那些“弱问句”
        prefix = q[:matches[best_idx].start()].strip("，。；;!！~～ ")
        if prefix and "？" in prefix:
            prefix = prefix.split("？", 1)[0].strip("，。；;!！~～ ")
        if prefix and not self._looks_like_question(prefix):
            return f"{prefix}，{chosen}？"
        return f"{chosen}？"

    def _normalize_for_similarity(self, text: str) -> str:
        import re
        t = (text or "").strip().lower()
        t = re.sub(r"\s+", "", t)
        t = re.sub(r"[，。！？、:：;；\"'“”‘’（）()【】\[\]~～\-—]", "", t)
        return t

    def _question_semantic_slot(self, text: str) -> str:
        normalized = self._normalize_for_similarity(text)
        if not normalized:
            return ""
        if any(keyword in normalized for keyword in ("坐着", "舒服", "累不累", "撑得住", "缓一缓")):
            return "current_comfort"
        return ""

    def _looks_like_location_answer_text(self, text: str) -> bool:
        import re

        raw = (text or "").strip()
        if not raw or self._looks_like_question(raw):
            return False
        return bool(re.search(r'(省|市|区|县|医院|楼|层)', raw))

    def _recent_history_has_slot(self, slot: str, conversation_history: Optional[List[Dict]], avoid_questions: Optional[List[str]] = None) -> bool:
        if not slot:
            return False

        history_items = conversation_history or []
        for msg in history_items[-8:]:
            if msg.get("role") != "assistant":
                continue
            if self._question_semantic_slot(msg.get("content", "")) == slot:
                return True

        for text in (avoid_questions or [])[-12:]:
            if self._question_semantic_slot(text) == slot:
                return True
        return False

    def _sanitize_location_followup_drift(
        self,
        utterance: str,
        last_user_msg: str,
        conversation_history: Optional[List[Dict]] = None,
        avoid_questions: Optional[List[str]] = None,
    ) -> str:
        text = (utterance or "").strip()
        return text

    def _looks_like_fixed_target_meta_lead(self, text: str) -> bool:
        import re

        raw = (text or "").strip()
        if not raw:
            return False

        patterns = [
            r"刚才聊了不少",
            r"刚才说了不少",
            r"聊了不少关于",
            r"关于(?:时间|地点|日期|星期|季节)",
            r"现在咱们聊聊",
            r"现在咱们说说",
            r"现在聊聊",
            r"接下来(?:咱们)?聊聊",
            r"咱们聊聊",
            r"咱们再说说",
            r"说到这里",
            r"咱们已经确认了",
            r"咱们已经知道",
            r"也聊了不少",
        ]
        return any(re.search(pattern, raw) for pattern in patterns)

    def _strip_fixed_target_meta_lead(self, text: str) -> str:
        import re

        cleaned = (text or "").strip()
        if not cleaned:
            return ""

        replacements = [
            r"咱们刚才聊了不少[^，。；;]*",
            r"刚才聊了不少[^，。；;]*",
            r"刚才说了不少[^，。；;]*",
            r"聊了不少关于[^，。；;]*",
            r"关于(?:时间|地点|日期|星期|季节)[^，。；;]*",
            r"现在咱们聊聊[^，。；;]*",
            r"现在咱们说说[^，。；;]*",
            r"现在聊聊[^，。；;]*",
            r"咱们已经确认了[^，。；;]*",
            r"咱们已经知道[^，。；;]*",
            r"也聊了不少[^，。；;]*",
            r"接下来(?:咱们)?聊聊[^，。；;]*",
            r"咱们聊聊[^，。；;]*",
            r"咱们再说说[^，。；;]*",
            r"说到这里[^，。；;]*",
        ]
        for pattern in replacements:
            cleaned = re.sub(pattern, "", cleaned)
        cleaned = re.sub(r"^[，,\s]+", "", cleaned)
        cleaned = re.sub(r"[，,\s]+$", "", cleaned)
        return cleaned.strip("，,。；;!！~～ ")

    def _get_location_target_by_task_id(self, task_id: Optional[str]) -> str:
        fixed_targets = {
            "orientation_place_province_city": "您现在这个医院是在什么省什么市呀？",
            "orientation_place_district": "您现在这个医院是在什么区或者县呀？",
            "orientation_place_location_floor": "您现在这个医院叫什么，在几楼呀？",
        }
        return fixed_targets.get((task_id or "").strip(), "")

    def _canonicalize_location_target_question(self, target_question: str, task_instruction: Optional[str] = None) -> str:
        target_text = (target_question or "").strip()
        instruction_text = (task_instruction or "").strip()
        combined_text = f"{instruction_text} {target_text}".strip()
        if not combined_text:
            return ""

        if not any(marker in combined_text for marker in ("省", "市", "区", "县", "楼", "楼层", "几楼", "地点", "地方", "医院")):
            return ""

        floor_markers = ("几楼", "楼层", "几层", "叫什么", "病房", "科室", "楼")
        district_markers = ("区", "县")

        if target_text:
            if any(marker in target_text for marker in floor_markers):
                return "您现在这个医院叫什么，在几楼呀？"
            if "省" in target_text and "市" in target_text:
                return "您现在这个医院是在什么省什么市呀？"
            if any(marker in target_text for marker in district_markers):
                return "您现在这个医院是在什么区或者县呀？"

        if any(marker in instruction_text for marker in floor_markers):
            return "您现在这个医院叫什么，在几楼呀？"
        if "省" in instruction_text and "市" in instruction_text:
            return "您现在这个医院是在什么省什么市呀？"
        if any(marker in instruction_text for marker in district_markers):
            return "您现在这个医院是在什么区或者县呀？"

        if any(marker in combined_text for marker in floor_markers):
            return "您现在这个医院叫什么，在几楼呀？"
        if "省" in combined_text and "市" in combined_text:
            return "您现在这个医院是在什么省什么市呀？"
        if any(marker in combined_text for marker in district_markers):
            return "您现在这个医院是在什么区或者县呀？"
        return ""

    def _sanitize_ack(self, ack: str, q: str = "") -> str:
        """清理 ack 中残留问句，并避免和 q 语义重复。"""
        import re
        from difflib import SequenceMatcher

        a = (ack or "").strip()
        if not a:
            return ""

        a = a.replace("?", "？")
        if "？" in a:
            a = a.split("？", 1)[0].strip()

        parts = [p.strip("，,。!！~～ ") for p in re.split(r"[。!！~～]", a) if p.strip("，,。!！~～ ")]
        kept_parts = []
        for p in parts:
            if self._looks_like_question(p):
                break
            kept_parts.append(p)

        if kept_parts:
            a = "。".join(kept_parts).strip("，,。 ")
        elif self._looks_like_question(a):
            a = ""
        else:
            a = a.strip("，,。 ")

        if a and q:
            na = self._normalize_for_similarity(a)
            nq = self._normalize_for_similarity(q)
            if na and nq:
                ratio = SequenceMatcher(None, na, nq).ratio()
                if na in nq or nq in na or ratio >= 0.82:
                    a = ""

        if a and not a.endswith(("。", "！", "!", "~", "～")):
            a += "。"
        return a

    def _compose_ack_with_target_question(self, ack: str, target_question: str) -> str:
        q = (target_question or "").strip()
        if not q:
            return self._sanitize_ack(ack, "")

        clean_ack = self._sanitize_ack(ack, q)
        if clean_ack:
            if clean_ack.endswith(("。", "！", "!", "~", "～")):
                return f"{clean_ack}{q}"
            return f"{clean_ack}，{q}"
        return q

    def _derive_target_question_core(self, target_question: str) -> str:
        import re

        text = (target_question or "").strip()
        if not text:
            return ""

        candidate = text.rstrip('，,；;。！!~～ ')
        sentence_parts = [seg.strip() for seg in re.split(r'[。！？!?；;]', candidate) if seg.strip()]
        if sentence_parts:
            candidate = sentence_parts[-1]

        comma_parts = [seg.strip() for seg in re.split(r'[，,]', candidate) if seg.strip()]
        if comma_parts:
            for seg in reversed(comma_parts):
                if self._looks_like_question(seg):
                    candidate = seg
                    break
            else:
                candidate = comma_parts[-1]

        candidate = re.sub(r'^(那|那就|那您|那你|对了|顺便|再|然后|所以)\s*', '', candidate)
        candidate = re.sub(r'^(我想问(?:下)?您?|想问(?:下)?您?|问(?:下)?您?|请问您?|麻烦问您?)\s*', '', candidate)
        candidate = candidate.strip().strip('，,；;。！!~～ ')
        if not candidate:
            return text
        if not candidate.endswith(("？", "?")) and self._looks_like_question(candidate):
            candidate = f"{candidate}？"
        return candidate

    def _matches_fixed_target(
        self,
        utterance: str,
        target_question: str,
        target_question_core: str = "",
    ) -> bool:
        surface = (target_question or "").strip()
        core = (target_question_core or "").strip() or self._derive_target_question_core(surface)
        return (
            (bool(surface) and self._ends_with_target_question(utterance, surface))
            or (bool(core) and self._ends_with_target_question(utterance, core))
        )

    def _extract_prefix_before_fixed_target(
        self,
        utterance: str,
        target_question: str,
        target_question_core: str = "",
    ) -> str:
        text = (utterance or "").strip()
        surface = (target_question or "").strip()
        core = (target_question_core or "").strip() or self._derive_target_question_core(surface)
        for needle in (surface, core):
            if needle and needle in text:
                idx = text.rfind(needle)
                return text[:idx].strip("，,。；;!！~～ ")
        return ""

    def _has_question_like_lead_before_fixed_target(
        self,
        utterance: str,
        target_question: str,
        target_question_core: str = "",
    ) -> bool:
        import re

        prefix = self._extract_prefix_before_fixed_target(utterance, target_question, target_question_core)
        if not prefix:
            return False
        segments = [seg.strip() for seg in re.split(r'[，,。；;]', prefix) if seg.strip()]
        tail_segment = segments[-1] if segments else prefix
        suspicious_patterns = [
            r"(是不是|有没有|还(会|能|是|算)?[^，。？?]{0,8}(吗|呀|啊)|舒服点|好点|对不对|行不行|还行吗|顺利吗|知道不知道|记不记得|想不想得起)",
        ]
        if self._looks_like_question(tail_segment):
            return True
        return any(re.search(pattern, tail_segment) for pattern in suspicious_patterns)

    def _extract_fixed_target_ack_source(
        self,
        utterance: str,
        target_question: str,
        target_question_core: str = "",
    ) -> str:
        import re

        text = (utterance or "").strip()
        if not text:
            return ""

        prefix = self._extract_prefix_before_fixed_target(utterance, target_question, target_question_core)
        if prefix:
            segments = [seg.strip() for seg in re.split(r'[，,。；;]', prefix) if seg.strip()]
            kept_segments = []
            for seg in segments:
                if self._looks_like_question(seg):
                    continue
                kept_segments.append(seg)
            if kept_segments:
                return "，".join(kept_segments).strip("，,。；;!！~～ ")
            return prefix.strip("，,。；;!！~～ ")

        return ""

    def _extract_fixed_target_tail(
        self,
        utterance: str,
        target_question: str,
        target_question_core: str = "",
    ) -> str:
        text = (utterance or "").strip()
        surface = (target_question or "").strip()
        core = (target_question_core or "").strip() or self._derive_target_question_core(surface)

        if not text:
            return surface or core
        if surface and text.endswith(surface):
            return surface
        if core:
            idx = text.rfind(core)
            if idx != -1:
                prefix = text[:idx]
                sentence_cut = max(prefix.rfind("。"), prefix.rfind("！"), prefix.rfind("!"), prefix.rfind("~"), prefix.rfind("～"))
                if sentence_cut != -1:
                    tail = text[sentence_cut + 1:].strip()
                    if tail:
                        return tail
                comma_cut = max(prefix.rfind("，"), prefix.rfind(","))
                if comma_cut != -1:
                    short_lead = prefix[comma_cut + 1:].strip()
                    if short_lead and len(short_lead) <= 8:
                        tail = text[comma_cut + 1:].strip()
                        if tail:
                            return tail
                return text[idx:].strip()
        return surface or core or text

    def _extract_unstreamed_fixed_target_tail(
        self,
        utterance: str,
        streamed_prefix: str,
        target_question: str,
        target_question_core: str = "",
    ) -> str:
        text = (utterance or "").strip()
        prefix = (streamed_prefix or "").strip()
        if not text:
            return self._extract_fixed_target_tail(text, target_question, target_question_core)
        if prefix:
            if text.startswith(prefix):
                remaining = text[len(prefix):].strip()
                if remaining:
                    return remaining
            normalized_prefix = self._normalize_for_similarity(prefix)
            normalized_text = self._normalize_for_similarity(text)
            if normalized_prefix and normalized_text.startswith(normalized_prefix):
                cut_idx = 0
                for idx in range(len(text)):
                    if len(self._normalize_for_similarity(text[:idx + 1])) >= len(normalized_prefix):
                        cut_idx = idx + 1
                        break
                if cut_idx and cut_idx < len(text):
                    remaining = text[cut_idx:].strip()
                    if remaining:
                        return remaining
        return self._extract_fixed_target_tail(text, target_question, target_question_core)

    def _ends_with_target_question(self, utterance: str, target_question: str) -> bool:
        text = self._normalize_for_similarity(utterance)
        q = self._normalize_for_similarity(target_question)
        if not text or not q:
            return False
        return text.endswith(q)

    def _is_target_question_fragment(self, text: str, target_question: str) -> bool:
        import re

        candidate = self._normalize_for_similarity(text)
        q = self._normalize_for_similarity(target_question)
        if not candidate or not q or len(candidate) < 4:
            return False

        if q.startswith(candidate) or q.endswith(candidate) or candidate in q:
            return True

        trimmed = re.sub(r'^(那|对了|然后|再|顺便)\s*', '', candidate)
        if trimmed and len(trimmed) >= 4 and (q.startswith(trimmed) or q.endswith(trimmed) or trimmed in q):
            return True
        return False

    def _resolve_fixed_target_utterance(
        self,
        utterance: str,
        target_question: str,
        target_question_core: str = "",
        last_user_msg: str = "",
    ) -> tuple[str, bool, str]:
        import re

        compose_target = (target_question or "").strip() or (target_question_core or "").strip()
        core_target = (target_question_core or "").strip() or self._derive_target_question_core(compose_target)

        text = self._strip_user_echo(utterance, last_user_msg).strip()
        if not text:
            return self._compose_ack_with_target_question("", compose_target), True, ""

        text = self._keep_single_question(text)
        prefix_before_target = self._extract_prefix_before_fixed_target(text, compose_target, core_target)
        has_meta_lead = self._looks_like_fixed_target_meta_lead(prefix_before_target)
        is_location_target = bool(self._canonicalize_location_target_question(compose_target))
        medical_markers = ("医院", "病房", "看病", "检查", "挂号", "来回", "排队", "折腾")
        long_unfocused_lead = (
            is_location_target
            and bool(prefix_before_target)
            and len(self._normalize_for_similarity(prefix_before_target)) > 28
            and not any(marker in prefix_before_target for marker in medical_markers)
        )
        if self._matches_fixed_target(text, compose_target, core_target):
            if not self._has_question_like_lead_before_fixed_target(text, compose_target, core_target) and not has_meta_lead and not long_unfocused_lead:
                return text, False, ""
            self._log_verbose("固定问题命中 target，但前缀仍像额外问句/主持词/过长铺垫，改走最小兜底", level="warn")

        ack_source = self._extract_fixed_target_ack_source(text, compose_target, core_target)
        if not ack_source:
            trailing_question = re.search(r'^(.*?)([^？?]*[？?])\s*$', text)
            if trailing_question:
                ack_source = trailing_question.group(1).strip("，。；;!！~～ ")
            else:
                ack_source = text if not self._looks_like_question(text) else ""

        if is_location_target and ack_source:
            ack_source = self._strip_fixed_target_meta_lead(ack_source)

        ack = self._sanitize_ack(ack_source, core_target or compose_target)
        if not ack and ack_source:
            ack = self._sanitize_ack(ack_source, "")
        if is_location_target and ack:
            ack = self._strip_fixed_target_meta_lead(ack)
            ack_segments = [seg.strip("，,。；;!！~～ ") for seg in re.split(r'[。；;]', ack) if seg.strip("，,。；;!！~～ ")]
            if ack_segments:
                ack = ack_segments[0]
            if len(self._normalize_for_similarity(ack)) > 20 and not any(marker in ack for marker in medical_markers):
                ack = ""

        debug_ack = ack or ack_source
        return self._compose_ack_with_target_question(ack, compose_target), True, debug_ack

    def _strip_user_echo(self, utterance: str, last_user_msg: str) -> str:
        from difflib import SequenceMatcher
        import re

        text = (utterance or "").strip()
        user_text = (last_user_msg or "").strip()
        if not text or not user_text:
            return text

        m = re.search(r'([^？?]*[？?])\s*$', text)
        if not m:
            return text

        prefix = text[:m.start(1)].strip("，。；;!！~～ ")
        q_part = m.group(1).strip()
        if not prefix or not q_part:
            return text

        nprefix = self._normalize_for_similarity(prefix)
        nuser = self._normalize_for_similarity(user_text)
        if not nprefix or not nuser:
            return text
        if len(nprefix) < 6:
            return text

        matcher = SequenceMatcher(None, nprefix, nuser)
        overlap = matcher.find_longest_match(0, len(nprefix), 0, len(nuser)).size
        ratio = matcher.ratio()
        min_len = min(len(nprefix), len(nuser))
        echoed_fragment = (nprefix in nuser or nuser in nprefix) and overlap >= max(6, int(min_len * 0.6))
        near_verbatim = ratio >= 0.88 or overlap >= max(8, int(min_len * 0.75))
        if echoed_fragment or near_verbatim:
            return q_part

        return text

    def _extract_topic_hint(self, task_instruction: Optional[str], bridge_hint: Optional[str]) -> str:
        import re
        task = task_instruction or ""
        m = re.search(r"「([^」]{1,12})」", task)
        if m:
            return m.group(1).strip()
        if bridge_hint:
            return bridge_hint.split("→")[-1].strip()
        return ""

    def _is_too_open_ended(self, q: str) -> bool:
        text = (q or "").strip()
        if not text:
            return False
        risky_patterns = [
            r"新鲜事",
            r"分享一下",
            r"最近.*(怎么样|咋样|如何)",
            r"最近有.*(吗|么|呢|？|\?)",
            r"讲讲",
            r"说说看",
        ]
        import re
        return any(re.search(p, text) for p in risky_patterns)

    def _rewrite_open_question(
        self, q: str, topic_hint: str, dimension_name: str
    ) -> str:
        """仅在命中空泛问法时触发，把问题改成更具体、老人更易回答的问法。"""
        topic = (topic_hint or "").strip()
        if "爱好" in topic or "兴趣" in topic:
            return "您平时更喜欢散散步，还是在家看看电视呀？"
        if "生活" in topic or "日常" in topic:
            return "您今天白天在家一般做点啥呀？"
        if "心情" in topic:
            return "您今天心情挺不错的，是不是和家里人聊了会儿天呀？"
        if "回忆" in topic:
            return "您年轻的时候平时最爱做什么呀？"
        if "天气" in topic:
            return "今天天气还行，您今天有出去走走吗？"
        if dimension_name == "闲聊":
            return "您今天在家是看看电视，还是听听歌呀？"
        return q

    def _polish_structured_question(self, question: str, task_instruction: Optional[str], must_include: Optional[List[str]]) -> str:
        import re

        text = (question or "").strip()
        if not text:
            return text

        instruction = task_instruction or ""
        must_text = " ".join(must_include or [])
        is_weekday_task = "星期几" in instruction or "周几" in instruction or "星期几" in must_text or "周几" in must_text
        is_season_task = "季节" in instruction or "季节" in must_text

        if not is_weekday_task and not is_season_task:
            return text

        if is_weekday_task:
            if re.search(r"(今天|今儿|现在).{0,6}(是)?星期吗", text) or re.search(r"知道今天是星期吗", text):
                text = re.sub(r"(今天|今儿|现在).{0,6}(是)?星期吗", "今天星期几来着", text)
                text = re.sub(r"知道今天是星期吗", "知道今天星期几吗", text)
            if "星期吗" in text:
                text = text.replace("星期吗", "星期几吗")
            if "知道今天是星期几吗" in text:
                text = text.replace("知道今天是星期几吗", "知道今天星期几吗")

        if is_season_task:
            text = re.sub(r"(什么季节)(\s*\1)+", r"\1", text)
            text = re.sub(r"什么季节\s*什么季", "什么季节", text)
            text = re.sub(r"季节\s*什么季节", "什么季节", text)
            text = re.sub(r"您知道现在是什么季节(?:呀|啊|呢)?", "您知道现在是什么季节吗", text)
            text = re.sub(r"现在是什么季节(?:呀|啊|呢)?", "现在是什么季节呀", text)

        return text

    def _rewrite_utterance_if_needed(
        self, utterance: str, topic_hint: str, dimension_name: str
    ) -> str:
        """对一体化 utterance 做轻量质量修正（仅在空泛问法时触发）。"""
        import re
        text = (utterance or "").strip()
        if not text or not self._is_too_open_ended(text):
            return text

        q_match = re.search(r'([^？?]*[？?])\s*$', text)
        if not q_match:
            return self._rewrite_open_question(text, topic_hint, dimension_name)

        q_part = q_match.group(1).strip()
        prefix = text[:q_match.start(1)].strip()
        new_q = self._rewrite_open_question(q_part, topic_hint, dimension_name)
        if prefix:
            prefix = prefix.rstrip('，。 ')
            return f"{prefix}。{new_q}"
        return new_q

    def _has_ack_before_question(self, utterance: str) -> bool:
        """判断是否满足“先回应，再提问”结构。"""
        import re

        text = (utterance or "").strip()
        if not text:
            return False

        m = re.search(r'([^？?]*[？?])\s*$', text)
        if not m:
            return False

        prefix = text[:m.start(1)].strip("，。；;!！~～ ")
        if len(prefix) < 4:
            return False

        # 过滤“只有称呼”的伪回应
        if re.fullmatch(r'[\u4e00-\u9fa5]{1,6}(爷爷|奶奶|叔叔|阿姨|先生|女士)?', prefix):
            return False
        return True

    def _enforce_ack_with_llm(
        self,
        utterance: str,
        last_user_msg: str,
        topic_hint: str,
        active_llm: ChatOpenAI,
    ) -> str:
        """
        缺少回应时，用 LLM 二次改写为“先回应再提问”。
        仅做改写，不走代码模板兜底。
        """
        try:
            prompt = (
                "把下面这句话改写成一条口语化回复，严格只输出 JSON：{\"utterance\":\"...\"}\n"
                "硬性要求：\n"
                "1) 必须先回应对方刚才的话，再提一个问题。\n"
                "2) 只能有一个问号。\n"
                "3) 回应必须和对方原话直接相关，不能空泛附和。\n"
                "4) 问题要围绕目标话题。\n\n"
                f"对方原话：{last_user_msg}\n"
                f"目标话题：{topic_hint or '当前任务'}\n"
                f"原句：{utterance}"
            )
            response = active_llm.invoke([{"role": "user", "content": prompt}])
            raw = response.content if hasattr(response, "content") else str(response)
            raw = (raw or "").strip()
            parsed = json.loads(raw)
            rewritten = (parsed.get("utterance") or "").strip()
            if rewritten and self._has_ack_before_question(rewritten):
                return rewritten
        except Exception as e:
            self._log_verbose(f"二次改写失败，保留原句: {e}", level="warn")

        return utterance

    def _parse_json_payload(self, raw_output: str) -> Optional[dict]:
        """尽量从模型输出中提取并解析 JSON 对象。"""
        import re
        import ast

        text = (raw_output or "").strip()
        if not text:
            return None

        # 去掉 markdown 代码块包裹
        if "```" in text:
            text = re.sub(r"```json?\s*", "", text)
            text = text.replace("```", "").strip()

        candidates: List[str] = [text]

        # 尝试截取最外层 JSON 对象
        left = text.find("{")
        right = text.rfind("}")
        if left != -1 and right != -1 and right > left:
            candidates.append(text[left:right + 1])

        # 去重并尝试解析
        seen = set()
        for cand in candidates:
            if not cand or cand in seen:
                continue
            seen.add(cand)

            normalized_variants = [
                cand,
                re.sub(r",\s*([}\]])", r"\1", cand),  # 修复尾逗号
            ]

            for variant in normalized_variants:
                try:
                    parsed = json.loads(variant)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass

                # 兼容 python 字典风格（单引号）
                try:
                    parsed = ast.literal_eval(variant)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass

        return None
    
    def _run(
        self,
        dimension_name: str,
        dimension_description: str = "",
        knowledge_context: str = "",
        patient_age: Optional[int] = None,
        patient_education: Optional[int] = None,
        patient_name: Optional[str] = None,
        patient_gender: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,  # 🔥 类型变更：支持结构化历史
        generated_questions: Optional[List[str]] = None,  # 🔥 新增：已生成问题列表
        patient_emotion: Optional[str] = None,
        task_instruction: Optional[str] = None,
        task_id: Optional[str] = None,
        persona_hooks: Optional[List[str]] = None,
        must_include: Optional[List[str]] = None,
        avoid_questions: Optional[List[str]] = None,
        bridge_hint: Optional[str] = None,  # 🔥 新增：自然过渡提示
        target_question: Optional[str] = None,
        target_question_core: Optional[str] = None,
        conversation_summary: Optional[str] = None,  # 🧠 对话前情摘要
    ) -> str:
        """
        生成问题
        
        Returns:
            JSON格式：{"success": true, "question": "生成的问题"}
        """
        import time
        from src.utils.tool_logger import ToolLogger
        
        logger = ToolLogger("QuestionGenTool")
        self._set_active_logger(logger)
        logger.start(
            维度=dimension_name,
            任务ID=task_id or "-",
            原始Target=(target_question[:30] if target_question else "-"),
            原始CoreTarget=(target_question_core[:30] if target_question_core else "-"),
            任务指令=task_instruction[:30] if task_instruction else "N/A"
        )
        _start_time = time.time()
        raw_target_question_text = (target_question or "").strip()
        raw_target_question_core_text = (target_question_core or "").strip()
        target_question_text = (target_question or "").strip()
        target_question_core_text = (target_question_core or "").strip()
        if target_question_text and not target_question_core_text:
            target_question_core_text = self._derive_target_question_core(target_question_text)
        if not target_question_text:
            target_question_text = target_question_core_text
        task_location_target = self._get_location_target_by_task_id(task_id)
        explicit_location_target = self._canonicalize_location_target_question(
            target_question_text or target_question_core_text,
            None,
        )
        instruction_location_target = ""
        if task_location_target or explicit_location_target:
            instruction_location_target = self._canonicalize_location_target_question("", task_instruction)
        has_explicit_target = bool(target_question_text)
        if has_explicit_target and not explicit_location_target:
            if task_location_target:
                _target_preview = target_question_text[:60] + ("..." if len(target_question_text) > 60 else "")
                self._log_info(
                    f"显式固定目标为非地点题，忽略地点task兜底: task='{task_id}', target='{_target_preview}'",
                    level="warn",
                )
        elif explicit_location_target:
            authoritative_location_target = task_location_target or instruction_location_target or explicit_location_target
            if authoritative_location_target != explicit_location_target:
                self._log_info(
                    f"地点固定目标冲突，按{'task_id' if task_location_target else '任务指令'}纠正: authority='{authoritative_location_target}', target='{explicit_location_target}'",
                    level="warn",
                )
            target_question_text = authoritative_location_target
            target_question_core_text = authoritative_location_target
        elif instruction_location_target:
            target_question_text = instruction_location_target
            target_question_core_text = instruction_location_target
        if (
            raw_target_question_text
            or raw_target_question_core_text
            or target_question_text
            or target_question_core_text
            or task_location_target
            or instruction_location_target
        ):
            self._log_info(
                "固定目标归一化: "
                f"raw_target='{(raw_target_question_text[:60] + ('...' if len(raw_target_question_text) > 60 else '')) if raw_target_question_text else '-'}', "
                f"raw_core='{(raw_target_question_core_text[:60] + ('...' if len(raw_target_question_core_text) > 60 else '')) if raw_target_question_core_text else '-'}', "
                f"normalized_target='{(target_question_text[:60] + ('...' if len(target_question_text) > 60 else '')) if target_question_text else '-'}', "
                f"normalized_core='{(target_question_core_text[:60] + ('...' if len(target_question_core_text) > 60 else '')) if target_question_core_text else '-'}'"
            )
        fixed_target_mode = bool(target_question_text)
        if fixed_target_mode:
            _target_preview = target_question_text[:60] + ("..." if len(target_question_text) > 60 else "")
            if target_question_core_text and target_question_core_text != target_question_text:
                _core_preview = target_question_core_text[:60] + ("..." if len(target_question_core_text) > 60 else "")
                self._log_info(
                    f"固定目标模式: len={len(target_question_text)}, target='{_target_preview}', core='{_core_preview}'"
                )
            else:
                self._log_info(
                    f"固定目标模式: len={len(target_question_text)}, target='{_target_preview}'"
                )
        
        if fixed_target_mode:
            system_prompt = (
                "你是在诊室里陪老人轻松聊天的医生助理，说话自然、耐心、专业。\n"
                "表面是在聊天，不要说测评、测试、考一考、筛查，也不要用生硬的问卷腔。\n\n"
                '严格只输出 JSON：{"utterance":"..."}\n\n'
                "系统已经预先选好了最后真正要问用户的那一句固定目标话术。你的任务不是重新设计任务，而是直接生成最终整句：前面先较详细地回应对方的话，中间再写出较完整的过渡，最后自然落到固定目标话术。\n"
                "规则：\n"
                "1) utterance 必须是能直接说出口的最终回复。如果固定目标本身是问句，就只保留这一个问题；如果固定目标本身是指令/任务话术，也必须保留它原本的任务语义，不能改成别的问题或别的任务。\n"
                "2) 先接住对方刚说的一个具体点，并把这个点往下展开一层，让回应更完整、更像真人聊天；但展开只能基于对方明确说过的内容或顺手的生活联想，不能编造没说过的具体事实。\n"
                "3) 中间过渡也必须写出来，不能一脚踩到固定目标问题；要让人听得出这句话是怎么从前面的回应顺过去的。\n"
                "4) 如果当前话题和固定目标跨度有点大，就把这个弯写完整：可以补一层感受、节奏变化、生活场景或时间感，再自然落到固定目标问题，但不要硬造第二个问题。\n"
                "5) 禁止伪双问句、连着问两件事、为了过渡先来一句『是不是……呀』『有没有……呀』再问真正的问题。\n"
                "6) 禁止空泛套话和万能过渡词，例如『真好』『挺好』『挺有意思』『听着就不错』『说到这个』『对了』『顺便问一下』；回应和过渡都要有实际内容。\n"
                "7) 禁止完整复述对方原话；禁止用『我』代替对方复述他做过的事；禁止编造共同经历。\n"
                "8) 如果系统给了较长的自然问法，优先保留它真正要问的核心；不要把完整姓名或生硬称呼硬塞进句子里，也不要在结尾前再叠一层请问式引导。\n"
                "9) 当前外显场景是：医生或护士把患者带到安静诊室/评估室坐下，让他和系统轻松聊几句。不要写成候诊、排队、来院路上，也不要追问病情。\n"
                "10) 如果固定目标问题是在问地点，必须明确锚定当前所在医院/诊室这一处手填地址；优先说『您现在这个医院……』或『咱们现在这个房间……』，绝不能改写成老家、住址、常住地、平时住哪。\n"
                "11) 除非对方或系统明确提到，否则禁止断定住院、病房、食堂、陪护、调理中、住了几天。\n"
                "12) 禁止主动说出年份、月份、日期、星期、季节等评估答案。\n"
                "13) 如果固定目标是命名图片、记三个词、回忆三个词、复述句子、口算这类特殊评估任务，必须保留任务里的关键对象/词语/数字/句子，不得替换成别的图片对象、别的数字、别的句子，也不得提前泄露答案。\n"
                "14) 口语化、像医院里当面聊天的自然交流；允许写得更完整，但不要像作文、主持串词或任务脚本。\n"
                "15) 不要输出解释、前后缀或 Markdown。\n\n"
                "好的例子：\n"
                "- 对方说『火锅』，固定目标问题是『今年是哪一年来着？』 →『火锅还是得慢慢吃，一边涮一边聊，一顿饭坐下来心里都热乎。日子这么一晃也过得快，今年是哪一年来着？』\n"
                "- 对方说『春季了吧』，固定目标问题是『今天星期几来着？』 →『像是开春了，这阵子白天慢慢长起来了，外头的风也没前阵子那么硬。最近这几天过得还挺快的，今天星期几来着？』\n"
                "- 对方说『我刚坐下』，固定目标问题是『您看现在是什么季节呀？』 →『刚坐下先缓一缓，咱们慢慢聊，不着急。外头这阵子天气变化挺明显的，您看现在是什么季节呀？』\n"
                "- 对方说『医生让我在这儿聊一会儿』，固定目标问题是『您现在这个医院是在什么区或者县呀？』 →『对，咱们就轻松聊几句，医生也在旁边。您现在这个医院是在什么区或者县呀？』\n"
                "- 对方说『有的时候会』，固定目标话术是『我给您看一张图片，请您说说这是什么？』 →『有时候出去走走也挺好，人活动一下精神会松快些。我给您看一张图片，请您说说这是什么？』\n"
                "坏的例子：\n"
                "- 『跟朋友围着吃火锅热闹又暖身，真好，今年是哪一年呀？』 ← 空泛夸赞太多，虽然变长了，但回应和过渡都不够实在\n"
                "- 『火锅是不是也挺香的呀，今年是哪一年来着？』 ← 前面又造了一个伪过渡问句\n"
                "- 『好的，今年是哪一年来着？』 ← 回应和过渡都太薄，没有把对方的话真正展开\n"
                "- 『张先生，您今天心情挺好的，那您知道今年是哪一年吗？』 ← 称呼和回应都太重，评估腔太明显\n"
                "- 『您平常住的省份和城市是哪里呢？』 ← 把当前位置题改成了常住地/住址语义，错误\n"
                "- 『咱们刚才聊了不少关于时间的细节，现在咱们聊聊你现在所在的地方，请问您现在所在的地方属于哪个区/县呢？』 ← 总结前文+宣布换话题，太像主持词\n"
                "- 『住院本来就是慢慢调理的。您今天中午吃的是医院食堂的饭吗？』 ← 对方只说来医院看病，却被擅自补成住院/食堂场景\n"
                "- 『我给您看一张图片，您猜猜这是水果还是交通工具？』 ← 把原本的命名图片任务改成了别的内容，错误"
            )
        else:
            system_prompt = (
                "你是在诊室里陪老人轻松聊天的医生助理，说话自然、耐心、专业。\n"
                "表面是在聊天，不要说测评、测试、考一考、筛查，也不要用生硬的问卷腔。\n\n"
                '严格只输出 JSON：{"utterance":"..."}\n\n'
                "规则：\n"
                "1) utterance 必须先对对方刚说的核心信息做具体回应，再自然过渡到提问。回应可以有1-2小句，要有信息量，可以做生活化联想或态度反应，但不要照搬整句。\n"
                "2) utterance 里必须包含且仅包含一个问题（只问一件事）。\n"
                "3) 必须执行任务指令；若有 must_include，问题里必须包含。\n"
                "4) 若有 bridge_hint，问题必须直接命中该目标话题。\n"
                "5) 对认知减退老人，一次只问一件事，优先日常具体问题。\n"
                "5) 要热情，要像人说话。\n"
                "6) 避免空泛问法，如「最近有什么新鲜事分享一下」。\n"
                "7) 可用轻量选择式问法（如「您更喜欢A还是B」），但不要连环追问。\n"
                "8) 不要输出任何解释、前后缀或 Markdown。\n"
                "9) 回应部分必须抓住对方原话里的一个具体点（如地名、活动、数字、人物、食物）展开，用自己的话回应，禁止「我听着挺好」「好的」这类不含信息量的空泛附和。\n"
                '11) 不要重复询问对方在"最近聊天记录"中已经明确回答过的信息。只追问真正缺失的项。\n'
                '12) 【关键】若【任务指令】要求问某个信息，但「对方刚才说」里已经直接包含了这个信息（如任务要问星期几，对方刚才说了「星期六」），则禁止再问同样的事，改为顺水推舟接下去聊或自然换话题。\n'
                "10) 若对方表达拒绝/低落，先接纳情绪，再轻柔转话题，禁止硬夸赞。\n"
                "13) 【场景约束·很重要】当前外显场景是医生/护士把患者带到安静诊室或评估室，让他和系统轻松聊几句；不是候诊、排队、来院路上，也不是正式宣布做题。\n"
                "14) 能聊的话题必须是患者此刻坐在诊室里容易回答、且不依赖额外假设的：坐着舒不舒服、中午吃没吃、昨晚睡得怎样、眼前天气、现在这个房间/楼层、医生或家属是否在旁边（仅当对方提到）。\n"
                "   不要问：怎么来的、路上堵不堵、排队久不久、医院食堂吃了什么、病房几楼、住了几天、护工在不在、恢复得怎么样。\n"
                "15) 如果最近聊天记录里已经问过坐着是否舒服、吃饭、睡眠、天气、房间/楼层，视为同一类当下状态信息；不要换个说法再问一次。\n"
                "16) 如果对方刚回答的是省、市、区、县、医院、楼层这类地点信息，默认这是在回答当前医院位置；不要接成老家、常住地、平时住哪，也不要顺手回到已经问过的路况或车程。\n\n"
                "【严禁编造、复述和泄露答案·最高优先级】：\n"
                "❌ 绝对禁止用「我」开头复述对方做过的事（对方说「我出门走了走」→禁止说「我今天出门走了走，挺好」）\n"
                "❌ 绝对禁止说话不像人说的，比如，张先生知道今天是星期几吗？正常人没有这样说话的\n"
                "❌ 绝对禁止编造个人经历（如「我也住5楼」「我之前去过那里」「我也喜欢」「我今天也…」）\n"
                "❌ 你没有住址、没有经历、没有偏好，不要假装和对方有共同点\n"
                "❌ 禁止完整复述对方原话（不要照搬整句，只提炼一个关键信息点，用自己的话具体回应）\n"
                "❌ 禁止主动说出对方还没提到的事实信息（年份、月份、日期、星期、季节等）——这些要留给对方自己说\n"
                  "【语气必须像医院里当面聊天的自然交流，禁止以下公式化写法】：\n"
                "❌ 禁止用「那」做过渡词（如「那您住在哪个城市呢」）\n"
                "❌ 禁止书面语/文艺腔（如「春天的气息确实越来越浓了」）\n"
                "❌ 禁止「[称呼]+[文艺回应]+那+[直接提问]」的模板句式\n"
                "❌ 禁止用「哎我想起来了」「嗨对了」「说到这个」「话说」等口头禅做万能过渡——这些词用多了比「那」还死板\n"
                "❌ 禁止回应和提问之间毫无逻辑关联地硬拼（如聊完吃饭突然问天气）\n\n"
                "【核心原则：回应→提问必须有逻辑链条】\n"
                "✅ 回应对方的话后，下一个问题必须从回应内容中自然生长出来，让人觉得'问这个很合理'\n"
                "✅ 逻辑链条举例：吃饭→吃了什么/谁准备的，刚坐下→坐着舒不舒服，天气→现在看着亮不亮/是什么季节，房间→医院/楼层/区县\n"
                "✅ 如果必须换话题（任务要求），从对方的话里找一个关联点自然带过去，不要用口头禅硬转\n"
                "✅ 最近聊天记录里已经问过同一类当下状态，就不要再换说法回头问同一件事\n"
                "✅ 用自然口语（如「是啊」「对」「这样啊」），但不要每句都加\n\n"
                "好的示例（注意回应→提问之间的逻辑关联）：\n"
                "- 对方说「春天了」→ 可不是嘛，外头暖和多了，您这边是哪个城市呀，暖和起来了吗？（春天→天气→地理位置）\n"
                "- 对方说「刚吃完饭」→ 吃饱了好，吃的啥呀？（吃饭→吃了啥，顺着问）\n"
                "- 对方说「西红柿炒鸡蛋」→ 那得配碗米饭才过瘾，您自己做的还是家里人做的？（菜→谁做的）\n"
                "- 对方说「我喜欢散步」→ 散步是个好习惯，活动活动身子也舒服。您这会儿坐着还舒服不？（散步→身体感觉→当前状态）\n"
                "- 对方说「今天天气不错挺晴朗」→ 晴天看着人心里都亮堂。您看现在外头像是什么季节呀？（天气→季节）\n"
                "- 对方说「我住5楼」→ 5楼不高不低正好，爬楼还是坐电梯呀？（5楼→怎么上去）\n"
                "- 对方说「北京市海淀区」→ 海淀区啊，这下我听明白您说的是哪边了。折腾到现在，您这会儿坐着还舒服不？（地点答复→当前状态）\n"
                "- 对方说「陪我来的是我女儿」→ 女儿陪着就踏实多了，她是专门请假过来的吗？（用户已明确提到陪同人）\n"
                "- 对方说「早上挂的号」→ 一大早就安排好了，也不容易。咱们现在坐下来慢慢聊，您这会儿累不累？（挂号→当前状态）\n"
                "- 对方说「昨晚睡得不太好」→ 昨晚没睡好，人就容易发乏。咱们慢慢来，您这会儿坐着还撑得住不？（睡眠→当前状态）\n"
                "坏的示例（禁止）：\n"
                "- 西红柿炒鸡蛋配米饭最香啦。说到这个，今天天气怎么样呀？ ← 吃饭和天气毫无逻辑关联+「说到这个」硬转\n"
                "- 散步活动活动筋骨。嗨对了，今天是星期几呀？ ← 散步和星期几无关+口头禅过渡\n"
                "- 晴朗的天儿晒着舒服呀。哎我想起来了，您除了吃饭还做了哪些事呀？ ← 「哎我想起来了」万能过渡\n"
                "- 张先生，春天的气息确实越来越浓了。那您现在这个地方是哪个城市呢？ ← 太书面+公式化\n"
                "- 是啊，海淀区啊，我之前在海淀区待过一阵子。 ← 编造经历\n"
                "- 您在家住5楼啊，我也是住5楼，跟您一个楼层呢。 ← 编造共同点\n"
                "- 您在北京的呃北京市海淀区啊。 ← 鹦鹉学舌+复述噪音词\n"
                "- 我今天出门走了走，挺好。 ← 用「我」复述对方的活动，禁止！\n"
                "- 是啊，散步看书刷新闻，日子过得挺踏实。 ← 把对方原话照搬回去，禁止！\n"
                "- 是啊，2026年3月22日，周日，春天了。 ← 复述日期+泄露答案（春天），严重禁止！\n"
                "- 能保持不错的状态就挺好，住院本来就是慢慢调理的。您今天中午吃的是医院食堂的饭吗？ ← 对方只说来医院看病，却被擅自补成住院/食堂场景\n"
                "- 您今天住院第几天了？ ← 用户没说住院，不能自己补设住院事实\n"
                "- 护工这会儿在不在您旁边？ ← 用户没提护工/陪护，不能自己补设人物\n"
                "- 刚住进病房还适应吗？ ← 用户没说住院或病房，不能擅自改成住院情境\n"
                "- 大连靠海，平时住那儿应该挺舒服的。您刚才来这儿的时候路上堵车吗？ ← 把当前医院位置误写成常住地，还把话题带回候诊/路上"
            )

        # 🔥 延迟初始化 Context Cache（首次调用时创建）
        if self._use_volcengine and not self._context_cache_tried and not fixed_target_mode:
            self._context_cache_tried = True
            try:
                ctx_id = create_volcengine_context_cache(
                    model=self._default_model,
                    system_prompt=system_prompt,
                )
                if ctx_id:
                    self._context_id = ctx_id
                    self._context_llm = get_volcengine_context_chat_openai(
                        context_id=ctx_id,
                        model=self._default_model,
                        temperature=self._llm_temperature,
                        max_tokens=self._llm_max_tokens,
                        timeout=self._llm_timeout,
                        streaming=True,
                    )
                    self._log_verbose(f"Context Cache 已启用: {ctx_id}")
            except Exception as e:
                self._log_verbose(f"Context Cache 初始化异常: {e}", level="warn")

        user_prompt_parts = []
        
        # 聊天话题引导（内部使用，不暴露给用户）
        topic_hints = {
            '定向力': '围绕时间、地点做自然提问（如星期、日期、所在区域）。',
            '即时记忆': '让对方记住并复述简短信息（词语/数字）。',
            '注意力与计算': '用生活化小算术或连续计算测试注意力。',
            '延迟回忆': '回问之前提过的信息，观察回忆情况。',
            '语言': '围绕命名、复述、理解指令进行简短提问。',
            '构图(临摹)': '用简单图形临摹相关引导语。'
        }
        hint = topic_hints.get(dimension_name, '随便聊聊')
        user_prompt_parts.append(f"聊天方向提示：{hint}")

        if task_instruction:
            user_prompt_parts.append(f"\n【本轮要做的事（只给你看）】：{task_instruction}")

        if fixed_target_mode:
            user_prompt_parts.append(f"\n【固定目标问题（自然说法）】这句是已经预先选好的最终提问：{target_question_text}")
            if target_question_core_text and self._normalize_for_similarity(target_question_core_text) != self._normalize_for_similarity(target_question_text):
                user_prompt_parts.append(f"\n【固定目标问题（核心要问）】最后真正要落到的是这件事：{target_question_core_text}")
            user_prompt_parts.append(
                "\n【当前任务】请直接生成一整句最终回复：先较详细地回应对方，再写出较完整的过渡，并在结尾落到这句固定目标问题。"
                "若自然说法本身已经顺口，优先整体保留；不要另起新问题，也不要把真正要问的内容改掉。"
                "即使对方刚说的话很短，也不要只接半句；回应和过渡都要写得更完整一些，但不能编造没说过的事实。"
                "如果固定目标和当前话题跨度大，就把这个弯写出来，让人听得出是怎么顺过去的；但整句仍然只能保留最后那一个问题。"
                "固定目标若是地点题，要明确落在对方当前所在医院，不要只说模糊的『这个地方』。"
                "请把首轮就写成最终可直接说出口的版本，不要拆成前后两截。"
            )

        if persona_hooks:
            hooks_text = '、'.join([h for h in persona_hooks if h])
            if hooks_text:
                user_prompt_parts.append(
                    f"\n【个性化钩子】只有在非常顺的时候，才轻轻带到其中一个：{hooks_text}。"
                    "如果会让句子变得生硬、像编的，宁可完全不要提。"
                )

        if must_include and not fixed_target_mode:
            must_text = '、'.join([m for m in must_include if m])
            if must_text:
                user_prompt_parts.append(f"\n【必须包含】生成的问题里必须出现这些关键词/数字：{must_text}")
        
        # 🆕 合并非核心话题去重逻辑，统统加入 avoid_questions
        all_avoid = []
        if avoid_questions:
            all_avoid.extend(avoid_questions)
        if generated_questions:
            all_avoid.extend(generated_questions)
        
        if all_avoid:
            import re
            core_questions = []
            for q in all_avoid[-8:]:
                match = re.search(r'[，。]?([^，。]*[吗呢啊？?])$', q)
                if match:
                    core_questions.append(match.group(1).strip())
                else:
                    core_questions.append((q.split('，')[-1] if '，' in q else q).strip())
            if core_questions:
                avoid_text = '；'.join(core_questions[:6])
                user_prompt_parts.append(
                    f"\n【禁止重复】以下问法已问过，避免同义复问：{avoid_text}"
                )
                user_prompt_parts.append(
                    "\n【重复判定】不要只换个说法重复同一类问题。"
                    "例如：坐着舒不舒服/累不累算同一类；"
                    "早饭/午饭/晚饭吃了没，算同一类；天气/季节/外头亮不亮算同一类。"
                    "这些都必须避开，换一个新的信息点再问。"
                )
        
        # 🔥 新增：自然过渡提示 - 融入 ack
        if bridge_hint:
            target_topic = bridge_hint.split("→")[-1].strip() if "→" in bridge_hint else bridge_hint
            if fixed_target_mode:
                user_prompt_parts.append(
                    f"\n【过渡提示】回应时尽量顺着「{target_topic}」这个点展开一层，把过渡写完整，再自然落到固定目标问题；过渡可以更详细，但不能额外再造第二个问题。"
                )
            else:
                user_prompt_parts.append(
                    f"\n【话题约束】目标话题：{target_topic}。q 必须直接问这个话题，不能偏题。"
                )
        
        # 添加称呼信息（根据性别和年龄生成正确称呼）
        if patient_name:
            # 根据性别确定称呼后缀
            try:
                _age = int(patient_age) if patient_age else 60
            except (ValueError, TypeError):
                _age = 60
            if patient_gender == '男':
                suffix = '先生'
            else:
                suffix = '女士'
            full_name = f"{patient_name}{suffix}"
            user_prompt_parts.append(
                f"\n【称呼建议】平时直接用‘您’就行。"
                f"如果确实需要称呼，偶尔用‘{suffix}’或‘{full_name}’都可以；"
                f"不要把完整姓名反复塞进句子里，也不要每句话都叫一次。"
            )
        
        # 如果是定向力维度，注入当前真实时间、地点和天气信息
        location_task_text = f"{task_instruction or ''} {target_question_text}"
        is_location_task = any(marker in location_task_text for marker in ("省", "市", "区", "县", "地点", "地方", "楼", "楼层"))
        if ("定向" in dimension_name or "orientation" in dimension_description.lower()) and (not fixed_target_mode or is_location_task):
            # 获取完整实时上下文（位置、时间、天气）
            context = get_realtime_context()
            time_info = context['time']
            location = context['location']
            weather = context['weather']
            location_parts = [
                location.get('province', ''),
                location.get('city', ''),
                location.get('district', ''),
                location.get('hospital', ''),
                location.get('department', ''),
                location.get('place', ''),
            ]
            location_text = " ".join(part for part in location_parts if part)
            floor_text = location.get('floor', '')
            
            realtime_info = (
                f"\n【当前真实信息，聊天时可以用到】\n"
                f"今天是{time_info['year']}年{time_info['month']}月{time_info['day']}日，星期{time_info['weekday']}，{time_info['season']}\n"
                f"手填地址：{location_text}\n"
                f"手填楼层：{floor_text}\n"
                f"天气：{weather.get('temperature', '')} {weather.get('weather', '')}\n"
            )
            user_prompt_parts.append(realtime_info)
            if is_location_task:
                user_prompt_parts.append(
                    "\n【地点题特别提醒】如果这一轮是在问地点相关内容，必须围绕系统记录的手填地址来问。"
                    "优先问这份手填地址对应的医院、省、市、区县、具体地点、楼层；不要改问老家、家庭住址、平时住哪或常去哪里。"
                    "如果固定目标问题本身是地点题，也必须保留『现在这个医院』『当前所在医院』的意思，不能改写成『平常住哪里』或只说模糊的『这个地方』。"
                )
        
        if patient_age:
            user_prompt_parts.append(f"对方{patient_age}岁")
        
        if patient_emotion and patient_emotion != 'neutral':
            emotion_map = {'happy': '心情不错', 'sad': '有点低落', 'angry': '有点烦躁', 'fear': '有点紧张'}
            user_prompt_parts.append(f"对方现在{emotion_map.get(patient_emotion, '心情一般')}")
        
        # 🧠 注入前情摘要（长期记忆）
        if conversation_summary:
            user_prompt_parts.append(f"\n【前情摘要】{conversation_summary}")
        
        last_user_msg = ""
        context_heavy_ack = False
        context_dependency_reason = "not_checked"
        debug_fixed_target_ack = ""
        if conversation_history:
            # 🔥 优先从结构化历史中提取最近对话，避免截断导致的信息丢失
            if isinstance(conversation_history, list):
                history_list = conversation_history
            else:
                history_list = []
            
            prev_ai_responses = []  # 收集之前AI的回复，用于防重复
            
            if history_list and len(history_list) > 0:
                # 倒序查找最后一句用户的话
                for msg in reversed(history_list):
                    if msg.get('role') == 'user':
                        last_user_msg = msg.get('content', '')
                        break
                
                # 收集AI之前的回复（用于风格防重复）
                for msg in history_list:
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        if content and len(content) > 5:
                            prev_ai_responses.append(content[:60])
                
                # 最近对话原文（短期记忆窗口，由MemoryTool控制条数）
                context_desc = "\n【最近聊天记录】\n"
                for msg in history_list:
                    role_zh = "你" if msg.get("role") == "assistant" else "对方"
                    content = msg.get("content", "")
                    context_desc += f"{role_zh}: {content}\n"
                user_prompt_parts.append(context_desc)
            
            if last_user_msg:
                self._log_verbose(f"用户上轮回答: {last_user_msg}")
                user_prompt_parts.append(f"🔔 对方刚才说：「{last_user_msg}」")
                if fixed_target_mode:
                    if any(marker in last_user_msg for marker in ("医院", "看病", "不舒服", "难受", "检查", "挂号")):
                        user_prompt_parts.append(
                            "\n【特别提醒】如果对方刚提到医院、看病或身体不舒服，只接住『医生在旁边、先坐稳慢慢聊』这个当下场景，"
                            "不要写跑医院、排队、来回折腾，也不要继续追问哪里不舒服。"
                        )
                    context_heavy_ack, context_dependency_reason = self._diagnose_context_dependency(last_user_msg)
                    if (
                        context_heavy_ack
                        and target_question_text
                        and self._should_use_fast_structured_model(dimension_name, task_instruction)
                        and (
                            context_dependency_reason.startswith("short_reply")
                            or context_dependency_reason.startswith("short_without_concrete")
                            or context_dependency_reason.startswith("vague_marker")
                        )
                    ):
                        self._log_verbose("结构化固定问题已预生成 target_question，短回复不再强制升级高质量模型")
                        context_heavy_ack = False
                        context_dependency_reason = f"demoted_for_structured_fixed_target({context_dependency_reason})"
                    self._log_info(
                        f"固定问题承接判定: context_heavy={context_heavy_ack}, reason={context_dependency_reason}"
                    )
                    user_prompt_parts.append(
                        "\n【承接原则】优先回应「对方刚才说」；但如果这句话很短、含糊、带指代，"
                        "不能只按字面接，必须结合【最近聊天记录】和【前情摘要】判断当前正在聊的主线。"
                        "先把这个具体点较详细地说开，再写出较完整的过渡，最后自然落到固定目标问题；"
                        "即使前面写得更完整，整句也仍然只能保留最后那一个问题。"
                    )
                    if context_heavy_ack:
                        user_prompt_parts.append(
                            "\n【这句高度依赖上下文】这句本身信息不够，请优先参考【最近聊天记录】和【前情摘要】里最近正在聊的具体事情，"
                            "从那个具体点继续接话，不要只围着这几个字打转；同时让固定目标问题落得自然。"
                        )
                
                # 🔥 风格防重复：告诉LLM之前用了什么风格
                if prev_ai_responses:
                    # 提取之前回复的开头模式，让LLM避开
                    prev_starts = []
                    for resp in prev_ai_responses[-3:]:  # 最近3条AI回复
                        # 取回复的前15个字作为风格标记
                        start = resp[:15].rstrip('，。！')
                        if start:
                            prev_starts.append(start)
                    if prev_starts:
                        starts_text = '」「'.join(prev_starts)
                        user_prompt_parts.append(
                            f"\n🚫【防重复】你之前的回复开头是：「{starts_text}」\n"
                            f"这次**必须**用完全不同的开头、不同的回应方式、不同的过渡手法！"
                        )
            else:
                user_prompt_parts.append(f"\n最近对话：\n{str(conversation_history)[-300:]}")
        else:
            self._log_verbose("没有对话历史传入", level="warn")
        
        if fixed_target_mode:
            user_prompt_parts.append("\n直接输出 JSON，只写 utterance：")
        else:
            user_prompt_parts.append("\n直接输出你的回复（必须先回应再提问）：")
        
        user_prompt = "\n".join(user_prompt_parts)
        
        try:
            active_llm, active_model, llm_route_reason = self._select_llm_with_reason(
                dimension_name,
                task_instruction=task_instruction,
                last_user_msg=last_user_msg,
                bridge_hint=bridge_hint,
                prefer_context_quality=context_heavy_ack,
                fixed_target_mode=fixed_target_mode,
                target_question=target_question_text,
            )
            self._log_info(
                f"模型路由: model={active_model}, reason={llm_route_reason}, fixed_target={fixed_target_mode}, context_heavy={context_heavy_ack}"
            )
            if active_model != self._default_model:
                self._log_verbose(f"非主模型路径: {active_model} (维度={dimension_name})")

            # 🔥 Context Cache 路径：主模型 + 有缓存 → 跳过 system message
            use_context = (
                self._context_llm is not None
                and self._context_id is not None
                and not fixed_target_mode
                and active_llm is self._llm  # 仅主模型走缓存，平衡/快速模型走普通路径
            )
            cache_blockers = []
            if self._context_llm is None or self._context_id is None:
                cache_blockers.append("cache_not_ready")
            if fixed_target_mode:
                cache_blockers.append("fixed_target_mode")
            if active_llm is not self._llm:
                cache_blockers.append("non_default_model")

            if use_context:
                messages = [{"role": "user", "content": user_prompt}]
                _invoke_llm = self._context_llm
                self._log_verbose(f"Context Cache 命中 ({self._context_id[:20]}...)")
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                _invoke_llm = active_llm
                if self._use_volcengine:
                    self._log_verbose(f"Context Cache 未使用: {','.join(cache_blockers) if cache_blockers else 'disabled'}")

            # 🔥 流式路径：on_sentence 回调存在时，用 llm.stream() 逐句推送
            on_sentence = getattr(self, '_on_sentence_cb', None)
            streamed_any = False
            streamed_text_parts = []
            stream_sink = on_sentence
            if on_sentence and callable(on_sentence):
                def _tracked_on_sentence(text):
                    nonlocal streamed_any
                    candidate = (text or "").strip()
                    if fixed_target_mode and candidate:
                        stream_target_text = target_question_text or target_question_core_text
                        if stream_target_text and self._is_target_question_fragment(candidate, stream_target_text):
                            return
                        if stream_target_text and stream_target_text in candidate:
                            candidate = candidate.split(stream_target_text, 1)[0].strip()
                            if candidate.endswith(("我想问您", "想问您", "问您", "顺便问一句", "对了", "那")):
                                return
                        elif "？" in candidate or "?" in candidate:
                            return
                        if not candidate:
                            return
                    if candidate:
                        streamed_any = True
                        streamed_text_parts.append(candidate)
                        on_sentence(candidate)

                stream_sink = _tracked_on_sentence
            
            try:
                if stream_sink and callable(stream_sink):
                    raw_output = self._invoke_streaming(
                        _invoke_llm, messages, stream_sink,
                        fallback_llm=active_llm if use_context else None,
                        fast_llm=self._fast_llm,
                        json_field="utterance",
                    )
                else:
                    response = _invoke_llm.invoke(messages)
                    if hasattr(response, "content"):
                        raw_output = response.content
                    else:
                        raw_output = str(response)
            except Exception as invoke_error:
                if use_context:
                    # Context Cache 调用失败，降级为普通模式重试
                    self._log_verbose(f"Context Cache 调用失败，降级普通模式: {invoke_error}", level="warn")
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                    try:
                        response = active_llm.invoke(messages)
                    except Exception as fallback_error:
                        if self._fast_llm is not None:
                            self._log_info(f"主模型也失败，回退快速模型: {fallback_error}", level="warn")
                            active_model = self._fast_model or self._default_model
                            response = self._fast_llm.invoke(messages)
                        else:
                            raise
                elif active_llm is self._llm and self._fast_llm is not None:
                    # 主模型失败时回退到快速模型，避免一次失败拖慢整轮对话
                    self._log_info(f"主模型调用失败，回退快速模型: {invoke_error}", level="warn")
                    active_model = self._fast_model or self._default_model
                    response = self._fast_llm.invoke(messages)
                else:
                    raise
                # 非流式回退路径
                if hasattr(response, "content"):
                    raw_output = response.content
                else:
                    raw_output = str(response)
            
            raw_output = raw_output.strip()
            # print(f"  ├── 📝 LLM 原始输出: {raw_output[:50]}...")
            
            # 🆕 解析 JSON 结构化输出
            import re
            import json as json_module
            
            question = None  # 最终结果
            
            # 清洗 JSON：去掉 markdown 代码块标记
            cleaned_output = raw_output
            if "```" in cleaned_output:
                cleaned_output = re.sub(r"```json?\s*", "", cleaned_output)
                cleaned_output = cleaned_output.replace("```", "").strip()

            should_stream_target_question = False

            # 尝试解析 JSON（容错）
            parsed = self._parse_json_payload(cleaned_output)
            if parsed is not None:
                topic_hint = self._extract_topic_hint(task_instruction, bridge_hint)
                utterance = (parsed.get("utterance") or parsed.get("final_utterance") or parsed.get("reply") or "").strip()

                if fixed_target_mode:
                    raw_ack = (parsed.get("ack") or "").strip()
                    if utterance:
                        question, used_compose_fallback, fixed_ack = self._resolve_fixed_target_utterance(
                            utterance,
                            target_question_text,
                            target_question_core_text,
                            last_user_msg,
                        )
                        if used_compose_fallback:
                            debug_fixed_target_ack = fixed_ack
                            self._log_info(
                                f"固定问题JSON未保住 target_question，回退为兜底拼接: utterance='{utterance[:40]}...'",
                                level="warn",
                            )
                        else:
                            self._log_verbose(
                                f"固定问题JSON解析成功: utterance='{utterance[:40]}...', target='{(target_question_text or '')[:24]}...'"
                            )
                        self._log_verbose(f"固定问题最终结果: '{question[:80]}{'...' if len(question) > 80 else ''}'")
                    elif raw_ack:
                        ack = self._sanitize_ack(raw_ack, target_question_core_text or target_question_text)
                        debug_fixed_target_ack = ack
                        question = self._compose_ack_with_target_question(ack, target_question_text or target_question_core_text)
                        self._log_info(f"固定问题路径收到旧 ack 协议，已兼容兜底: ack='{ack[:40]}...'", level="warn")
                        self._log_verbose(f"固定问题最终结果: '{question[:80]}{'...' if len(question) > 80 else ''}'")
                    else:
                        question = self._compose_ack_with_target_question("", target_question_text or target_question_core_text)
                        self._log_verbose(f"固定问题最终结果: '{question[:80]}{'...' if len(question) > 80 else ''}'")
                elif utterance:
                    utterance = self._strip_user_echo(utterance, last_user_msg)
                    question = utterance
                    self._log_verbose(f"JSON解析成功: utterance='{utterance[:40]}...'")
                else:
                    # 兼容旧协议：ack + q
                    raw_ack = parsed.get("ack", "").strip()
                    q = parsed.get("q", "").strip()
                    if self._is_too_open_ended(q):
                        q = self._rewrite_open_question(q, topic_hint, dimension_name)
                    ack = self._sanitize_ack(raw_ack, q)
                    # 不使用代码兜底 ACK；若去重后为空但 LLM 提供了 ack，保留其清洗版
                    if not ack and raw_ack:
                        ack = self._sanitize_ack(raw_ack, "")
                    
                    self._log_verbose(f"JSON解析成功: ack='{ack}', q='{q[:30]}...'")
                    
                    # 🆕 校验：ack 不应该过分长（放宽到80字，允许更丰富的回应）
                    if ack and len(ack) > 80:
                        # 尝试在句号/感叹号处截断，保持语义完整
                        for i in range(60, len(ack)):
                            if ack[i] in '。！~':
                                ack = ack[:i+1]
                                break
                        else:
                            ack = ack[:70]
                    
                    # 拼接 acknowledgment 和 question
                    if ack and q:
                        # 根据 ack 结尾决定连接符
                        if ack.endswith(("！", "!", "~", "～", "。", ".")):
                            question = f"{ack}{q}"
                        else:
                            question = f"{ack}，{q}"
                    elif q:
                        question = q
                    elif ack:
                        question = ack
            else:
                self._log_verbose("JSON解析失败，尝试提取...", level="warn")
                ack_match = re.search(r'"ack"\s*:\s*"((?:\\.|[^"\\])*)"', cleaned_output, re.DOTALL)
                utterance_match = re.search(r'"utterance"\s*:\s*"((?:\\.|[^"\\])*)"', cleaned_output, re.DOTALL)
                if not utterance_match:
                    utterance_match = re.search(r'"final_utterance"\s*:\s*"((?:\\.|[^"\\])*)"', cleaned_output, re.DOTALL)
                if not utterance_match:
                    utterance_match = re.search(r'"reply"\s*:\s*"((?:\\.|[^"\\])*)"', cleaned_output, re.DOTALL)

                if utterance_match:
                    topic_hint = self._extract_topic_hint(task_instruction, bridge_hint)
                    utterance_raw = utterance_match.group(1).strip()
                    try:
                        utterance = json_module.loads(f"\"{utterance_raw}\"")
                    except Exception:
                        utterance = utterance_raw.replace('\\"', '"')
                    utterance = utterance.strip()
                    if fixed_target_mode:
                        question, used_compose_fallback, fixed_ack = self._resolve_fixed_target_utterance(
                            utterance,
                            target_question_text,
                            target_question_core_text,
                            last_user_msg,
                        )
                        if used_compose_fallback:
                            debug_fixed_target_ack = fixed_ack
                            self._log_info(
                                f"固定问题正则提取未保住 target_question，回退为兜底拼接: utterance='{utterance[:40]}...'",
                                level="warn",
                            )
                        else:
                            self._log_verbose(f"固定问题正则提取成功: utterance='{utterance[:40]}...'")
                        self._log_verbose(f"固定问题最终结果: '{question[:80]}{'...' if len(question) > 80 else ''}'")
                    else:
                        utterance = self._strip_user_echo(utterance, last_user_msg)
                        utterance = self._rewrite_utterance_if_needed(utterance, topic_hint, dimension_name)
                        self._log_verbose(f"正则提取成功: utterance='{utterance[:40]}...'")
                        question = utterance
                elif fixed_target_mode and ack_match:
                    raw_ack = ack_match.group(1).strip()
                    try:
                        raw_ack = json_module.loads(f"\"{raw_ack}\"")
                    except Exception:
                        raw_ack = raw_ack.replace('\\"', '"')
                    ack = self._sanitize_ack(raw_ack, target_question_core_text or target_question_text)
                    debug_fixed_target_ack = ack
                    question = self._compose_ack_with_target_question(ack, target_question_text or target_question_core_text)
                    self._log_info(f"固定问题正则命中旧 ack 协议，已兼容兜底: ack='{ack[:40]}...'", level="warn")
                    self._log_verbose(f"固定问题最终结果: '{question[:80]}{'...' if len(question) > 80 else ''}'")
                else:
                    # 尝试旧协议：ack + q
                    q_match = re.search(r'"q"\s*:\s*"((?:\\.|[^"\\])*)"', cleaned_output, re.DOTALL)
                    
                    if ack_match and q_match:
                        raw_ack = ack_match.group(1).strip()
                        q = q_match.group(1).strip()
                        try:
                            raw_ack = json_module.loads(f"\"{raw_ack}\"")
                        except Exception:
                            raw_ack = raw_ack.replace('\\"', '"')
                        try:
                            q = json_module.loads(f"\"{q}\"")
                        except Exception:
                            q = q.replace('\\"', '"')
                        topic_hint = self._extract_topic_hint(task_instruction, bridge_hint)
                        if self._is_too_open_ended(q):
                            q = self._rewrite_open_question(q, topic_hint, dimension_name)
                        ack = self._sanitize_ack(raw_ack, q)
                        # 不使用代码兜底 ACK；若去重后为空但 LLM 提供了 ack，保留其清洗版
                        if not ack and raw_ack:
                            ack = self._sanitize_ack(raw_ack, "")
                        self._log_verbose(f"正则提取成功: ack='{ack}', q='{q[:30]}...'")
                        if ack and q:
                            question = f"{ack}，{q}" if not ack.endswith(("！", "!", "~", "～")) else f"{ack}{q}"
                        else:
                            question = q or ack
                    else:
                        # JSON 完全失败，回退到原始输出
                        self._log_info("JSON 完全失败，回退到原始输出", level="warn")
                        question = raw_output.strip().strip('"').strip("'")
            
            # 如果还是没有问题，使用回退
            if not question or len(question) < 3:
                question = raw_output.strip().strip('"').strip("'")
            
            # 最终清洗
            question_before_cleanup = question
            question = re.sub(r"^(医生|护士|我)[:：]\s*", "", question).strip()
            question = self._strip_user_echo(question, last_user_msg)
            if fixed_target_mode and target_question_text:
                question = self._keep_single_question(question)
            else:
                question = self._sanitize_location_followup_drift(
                    question,
                    last_user_msg,
                    conversation_history,
                    avoid_questions,
                )
                question = self._polish_structured_question(question, task_instruction, must_include)
                if streamed_any and streamed_text_parts:
                    streamed_question = "".join(streamed_text_parts).strip()
                    streamed_question = re.sub(r"^(医生|护士|我)[:：]\s*", "", streamed_question).strip()
                    streamed_question = self._strip_user_echo(streamed_question, last_user_msg)
                    streamed_question = self._sanitize_location_followup_drift(
                        streamed_question,
                        last_user_msg,
                        conversation_history,
                        avoid_questions,
                    )
                    streamed_question = self._polish_structured_question(streamed_question, task_instruction, must_include)
                    if streamed_question and self._normalize_for_similarity(streamed_question) != self._normalize_for_similarity(question):
                        self._log_info(
                            f"流式文本与最终问题不一致，采用流式文本: streamed='{streamed_question[:40]}...', final='{question[:40]}...'",
                            level="warn",
                        )
                        question = streamed_question

            if fixed_target_mode and (target_question_text or target_question_core_text):
                if question:
                    should_stream_target_question = True
                if (
                    not self._matches_fixed_target(question, target_question_text, target_question_core_text)
                    or self._has_question_like_lead_before_fixed_target(question, target_question_text, target_question_core_text)
                ):
                    question, repaired_with_compose, repaired_ack = self._resolve_fixed_target_utterance(
                        question,
                        target_question_text,
                        target_question_core_text,
                        "",
                    )
                    if repaired_with_compose and repaired_ack:
                        debug_fixed_target_ack = debug_fixed_target_ack or repaired_ack
                _before_preview = question_before_cleanup[:80] + ("..." if len(question_before_cleanup) > 80 else "")
                _after_preview = question[:80] + ("..." if len(question) > 80 else "")
                self._log_verbose(f"固定问题后处理: before='{_before_preview}', after='{_after_preview}'")
                if debug_fixed_target_ack and self._normalize_for_similarity(question) == self._normalize_for_similarity(target_question_text or target_question_core_text):
                    self._log_verbose(
                        f"最终结果退化为纯 target_question，ack 可能被后处理折叠: ack='{debug_fixed_target_ack[:40]}...'",
                        level="warn",
                    )

            if fixed_target_mode and should_stream_target_question and on_sentence and callable(on_sentence):
                try:
                    tail_text = self._extract_unstreamed_fixed_target_tail(
                        question,
                        "".join(streamed_text_parts),
                        target_question_text,
                        target_question_core_text,
                    ) if streamed_any else question
                    if tail_text:
                        self._log_verbose(
                            f"固定问题补发尾句: streamed_any={streamed_any}, tail='{tail_text[:60]}{'...' if len(tail_text) > 60 else ''}'"
                        )
                        on_sentence(tail_text)
                except Exception:
                    pass

            # 清理问题末尾
            if question and not question.endswith(("？", "?", "。", ".", "！", "!", "~", "～")):
                question += "？"
            
            result = json.dumps({
                "success": True,
                "question": question,
                "dimension": dimension_name
            }, ensure_ascii=False, indent=2)
            
            return self._finalize_result(question, dimension_name, _start_time)
            
        except Exception as e:
            _elapsed = time.time() - _start_time
            self._log_info(f"问题生成失败 (耗时: {_elapsed:.2f}秒): {e}", level="error")
            active_logger = self._get_active_logger()
            if active_logger is not None:
                active_logger.end(error=str(e)[:60], elapsed=f"{_elapsed:.2f}s")
            return json.dumps({
                "success": False,
                "error": str(e),
                "fallback_question": f"请您描述一下您的{dimension_name}情况。"
            }, ensure_ascii=False, indent=2)
        finally:
            self._clear_active_logger()
    
        

    

            
    def _invoke_streaming(self, llm, messages, on_sentence, fallback_llm=None, fast_llm=None, json_field="utterance"):
        """
        流式调用 LLM，从 JSON 流中提取指定字段文本，按句回调。
        
        LLM 输出格式: {"utterance":"句子1。句子2？"} / {"ack":"句子1。"}
        流式提取字段值，每遇到句末标点就通过 on_sentence 回调推送。
        返回完整的 raw_output 供下游 JSON 解析使用。
        """
        import re as _re
        import time as _time
        
        full_text = ""
        utterance_started = False
        sentence_buf = ""
        first_sentence_time = None
        t0 = _time.time()
        first_sentence_sent = False
        
        FIRST_SENTENCE_ENDS = set('，。！？!?,')
        LATER_SENTENCE_ENDS = set('。！？!?')

        def _emit_stream_sentence(text: str):
            nonlocal first_sentence_time, first_sentence_sent
            candidate = (text or "").strip()
            if not candidate:
                return
            if first_sentence_time is None:
                first_sentence_time = _time.time() - t0
            try:
                on_sentence(candidate)
            except Exception:
                pass
            first_sentence_sent = True

        def _drain_sentence_buffer():
            nonlocal sentence_buf
            while True:
                current_ends = FIRST_SENTENCE_ENDS if not first_sentence_sent else LATER_SENTENCE_ENDS
                boundary_idx = -1
                for i, c in enumerate(sentence_buf):
                    if c in current_ends:
                        boundary_idx = i
                        break
                if boundary_idx < 0:
                    return
                _emit_stream_sentence(sentence_buf[:boundary_idx + 1])
                sentence_buf = sentence_buf[boundary_idx + 1:]
        
        try:
            for chunk in llm.stream(messages):
                token = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if not token:
                    continue
                full_text += token
                
                if not utterance_started:
                    # 检测目标 JSON 字段值的起始引号
                    match = _re.search(rf'"{_re.escape(json_field)}"\s*:\s*"', full_text)
                    if match:
                        utterance_started = True
                        # 提取引号后已有的文本
                        sentence_buf = full_text[match.end():]
                        _drain_sentence_buffer()
                else:
                    sentence_buf += token
                    _drain_sentence_buffer()
                    
                    # 检测 utterance 结束（未转义的引号）
                    if '"' in token and not sentence_buf.rstrip('"').endswith('\\'):
                        # 去掉结尾引号，推送剩余文本
                        remaining = sentence_buf.split('"', 1)[0]
                        if remaining.strip():
                            _emit_stream_sentence(remaining)
                        sentence_buf = ""
                        break
            
            # 推送缓冲区中剩余文本
            if sentence_buf.strip() and sentence_buf.strip() not in ('"', '"}', '}'):
                clean = sentence_buf.strip().rstrip('"}').strip()
                if clean:
                    _emit_stream_sentence(clean)
            
            elapsed = _time.time() - t0
            if first_sentence_time:
                self._log_verbose(f"流式完成: 总耗时={elapsed:.2f}s, 首句延迟={first_sentence_time:.2f}s")
            else:
                self._log_verbose(f"流式完成: 总耗时={elapsed:.2f}s")
            return full_text.strip()
            
        except Exception as e:
            self._log_info(f"流式调用失败({e})，降级为 invoke", level="warn")
            # 降级为非流式
            try_llm = fallback_llm or llm
            response = try_llm.invoke(messages)
            raw = response.content if hasattr(response, 'content') else str(response)
            return raw.strip()
    
    @staticmethod
    def _flush_sentences(text, on_sentence, ends):
        """从 text 中提取完整句子并推送"""
        pos = 0
        for i, c in enumerate(text):
            if c in ends:
                sentence = text[pos:i + 1].strip()
                if sentence:
                    try:
                        on_sentence(sentence)
                    except Exception:
                        pass
                pos = i + 1

    def _finalize_result(self, question, dimension_name, start_time):
        """统一处理最终结果：清理符号、日志输出、构建JSON"""
        import re
        import json as json_module
        import time
        
        # 最终清洗
        question = re.sub(r"^(医生|护士|我)[:：]\s*", "", question).strip()
        
        # 🔥 清理 Markdown 符号 ** 和多余逗号
        question = question.replace("**", "")  # 移除 **
        question = re.sub(r"，+", "，", question)  # 多个逗号合并为一个
        question = re.sub(r"。，", "，", question)  # 。，变为，
        question = re.sub(r"，(?=[？?。!！~～])", "", question)  # 逗号后面直接跟标点时删除逗号
        
        # 清理问题末尾
        if question and not question.endswith(("？", "?", "。", ".", "！", "!", "~", "～")):
            question += "？"
        
        result = json_module.dumps({
            "success": True,
            "question": question,
            "dimension": dimension_name
        }, ensure_ascii=False, indent=2)
        
        logger = self._get_active_logger()
        if logger is not None:
            logger.end(生成问题=question[:50])
        return result

    def generate_natural_transition(
        self,
        user_answer: str,
        dimension_name: str,
        patient_name: Optional[str] = None,
        patient_gender: Optional[str] = None,
        patient_age: Optional[int] = None,
        chat_history: Optional[List[Dict]] = None,
        current_emotion: str = 'neutral',
    ) -> str:
        """
        生成自然过渡回应（回应老人的话 + 自然引出下一个话题）
        
        Args:
            user_answer: 用户的回答
            dimension_name: 当前评估维度
            patient_name: 患者姓名
            patient_gender: 患者性别
            patient_age: 患者年龄
            chat_history: 对话历史
            current_emotion: 当前情绪
            
        Returns:
            JSON格式：{"success": true, "transition": "过渡回应"}
        """
        import time
        import random
        _start = time.time()
        
        # 生成称呼
        if patient_name:
            try:
                _age = int(patient_age) if patient_age else 60
            except (ValueError, TypeError):
                _age = 60
            if patient_gender == '男':
                suffix = '先生'
            else:
                suffix = '女士'
            greeting = f"{patient_name}{suffix}"
        else:
            greeting = ""
        
        try:
            # 提取最近对话
            recent_context = ""
            if chat_history and len(chat_history) >= 2:
                recent_context = "\n".join(recent_turns)
            
            system_prompt = (
                "你是在诊室里陪老人轻松聊天的医生助理。\n"
                "现在需要生成一个自然的过渡回应，让对话像医生在旁边陪着随口聊天。\n\n"
                "要求：\n"
                "1. **必须先具体回应对方刚才说的话！**（比如对方说吃了饺子，你要评两句饺子）\n"
                "2. 然后再用一句话自然地引出下一个话题或问题\n"
                "3. 语气亲切、口语化，但保持专业和分寸\n"
                "4. 控制在40-60字之间\n"
                "5. **禁止**生硬转折（如'说到这个'、'提到这个'）\n"
                "6. 不要带任何前缀（如'医生:'）\n"
            )
            
            user_prompt = f"""最近对话：
{recent_context}

对方刚才说："{user_answer}"
当前情绪：{current_emotion}
下一个话题方向：{dimension_name}

请生成一个自然的过渡回应（先具体回应对方说的"{user_answer}"，再自然过渡到"{dimension_name}")："""
            
            transition_llm, transition_model = self._select_llm(dimension_name)
            if transition_model != self._default_model:
                self._log_verbose(f"自然过渡使用非主模型: {transition_model}")

            response = transition_llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            # 提取文本
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            result = result.strip().strip('"').strip("'")
            
            # 清理前缀
            import re
            result = re.sub(r"^(医生|护士|我)[:：]\s*", "", result)
            
            # 添加称呼
            if greeting and not result.startswith(patient_name):
                result = f"{greeting}，{result}"
            
            self._log_verbose(f"自然过渡生成完成 (耗时: {(time.time()-_start)*1000:.0f}ms)")
            
            return json.dumps({
                "success": True,
                "transition": result
            }, ensure_ascii=False)
            
        except Exception as e:
            self._log_info(f"自然过渡生成失败: {e}", level="warn")
            # 备用方案
            responses = [
                "嗯，您说得对。对了，我想问您...",
                "理解理解。那咱们聊点别的...",
                "是这样啊。诶，我突然想到...",
            ]
            fallback = random.choice(responses)
            if greeting:
                fallback = f"{greeting}，{fallback}"
            
            return json.dumps({
                "success": True,
                "transition": fallback
            }, ensure_ascii=False)
    
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")
