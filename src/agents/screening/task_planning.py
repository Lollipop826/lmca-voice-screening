from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Generator, Set
import os
import json
import time
import re
import random
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

from src.agents.screening.catalog import ScreeningTaskCatalog
from src.agents.screening.state import ScreeningSessionState
from src.domain.dimensions import MMSE_DIMENSIONS
from src.utils.tool_logger import set_current_tool_log_session, log_summary


TASK_SELECTION_DESCRIPTIONS = {
    "persona_collect_1": "了解兴趣爱好",
    "persona_collect_2": "了解生活习惯",
    "orientation_time_year": "聊今年是哪一年",
    "orientation_time_season": "聊现在什么季节",
    "orientation_time_month_date": "聊几月几号",
    "orientation_time_weekday": "聊今天星期几",
    "orientation_place_province_city": "聊手填地址所在省/市",
    "orientation_place_district": "聊手填地址所在区/县",
    "orientation_place_location_floor": "聊手填地址对应地点/楼层",
    "registration_3words": "记3个词",
    "recall_3words": "回忆3个词",
    "attention_calc_life_math": "简单算术",
    "attention_reverse_phrase": "倒着说词组",
    "language_naming_watch": "命名(表)",
    "language_naming_pencil": "命名(笔)",
    "language_repetition_sentence": "复述句子",
    "language_reading_close_eyes": "读字动作",
    "language_3step_action": "三步指令",
    "language_writing_sentence": "说句子",
    "copy_pentagons": "临摹五边形",
}


@dataclass(frozen=True)
class TaskSelectionContext:
    non_buffer_candidates: list[str]
    forced_candidates: list[str]
    recent_chat: str
    used_categories: str
    max_buffer_rounds: int


class ScreeningTaskPlanning:
    """Select screening tasks using explicit state and service dependencies."""

    def __init__(
        self,
        *,
        use_local: bool,
        state: ScreeningSessionState,
        catalog: ScreeningTaskCatalog,
        memory_tool_provider: Callable[[], Any],
        classify_question: Callable[[str, List[str]], str],
        log_summary: Callable[[str, Dict[str, Any]], None],
        log_verbose: Callable[[str], None],
    ) -> None:
        self.use_local = use_local
        self.state = state
        self.catalog = catalog
        self._memory_tool_provider = memory_tool_provider
        self._classify_question = classify_question
        self._log_summary_sink = log_summary
        self._log_verbose_sink = log_verbose

    def _log_summary_card(self, title: str, items: Dict[str, Any]) -> None:
        self._log_summary_sink(title, items)

    def _log_verbose(self, message: str) -> None:
        self._log_verbose_sink(message)

    def classify_question_sync(self, question: str, candidates: List[str]) -> str:
        return self._classify_question(question, candidates)

    def _map_bridge_hint_to_task(self, bridge_hint: Optional[str], candidates: List[str]) -> Optional[str]:
        """
        将 LLM 选出的过渡话题直接映射到具体任务，避免已经命中评估点后又回到 buffer_chat。
        """
        if not bridge_hint or not candidates:
            return None

        topic_text = bridge_hint.split("→")[-1].strip() if "→" in bridge_hint else bridge_hint.strip()
        if not topic_text:
            return None

        matched_task = self.classify_question_sync(topic_text, candidates)
        if matched_task != "buffer_chat":
            self._log_summary_card("Task Routing", {"reason": "bridge_hint_direct", "topic": topic_text, "task": matched_task})
            return matched_task
        return None
    def _get_max_consecutive_buffer_chat(self) -> int:
        """读取连续 buffer 上限，默认 2（认知任务后最多闲聊2轮缓解情绪，然后强制进入下一个评估）。
        
        🔥 v1.5: 从 1 恢复为 2。闲聊2轮能有效缓解患者紧张情绪，
        同时 bridge_hint 保证过渡自然。
        """
        try:
            limit = int(os.getenv("MAX_CONSECUTIVE_BUFFER_CHAT", "2"))
        except ValueError:
            limit = 2
        return max(0, limit)
    def _get_valid_next_task_candidates(self, now: Optional[float] = None) -> List[str]:
        """计算当前状态下合法的下一任务候选（不含 buffer）。"""
        now = now or time.time()
        candidates = []

        basic_prereq = {
            "persona_collect_1",
            "persona_collect_2",
            "orientation_time_year",
            "orientation_time_season",
            "orientation_time_month_date",
            "orientation_time_weekday",
            "orientation_place_province_city",
            "orientation_place_district",
            "orientation_place_location_floor",
        }
        advanced_tasks = {
            "attention_calc_life_math",
            "attention_reverse_phrase",
            "registration_3words",
            "recall_3words",
            "language_naming_watch",
            "language_naming_pencil",
            "language_repetition_sentence",
            "language_reading_close_eyes",
            "language_3step_action",
            "language_writing_sentence",
            "copy_pentagons",
        }

        for task_id in self.catalog.required_tasks:
            if task_id in self.catalog.visual_action_tasks:
                continue
            if task_id in self.state._task_done:
                continue

            until_turn = self.state._task_cooldown_until.get(task_id)
            if until_turn is not None and self.state._turn_counter < until_turn:
                continue

            if task_id == "recall_3words":
                if "registration_3words" not in self.state._task_done:
                    continue
                if self.state._registration_ts is None:
                    continue
                elapsed = now - self.state._registration_ts
                if elapsed < 120:
                    continue

            if task_id == "registration_3words":
                if "persona_collect_1" not in self.state._task_done or "persona_collect_2" not in self.state._task_done:
                    continue

            candidates.append(task_id)

        if not basic_prereq.issubset(self.state._task_done):
            filtered = [t for t in candidates if t not in advanced_tasks]
            if filtered:
                candidates = filtered

        return candidates
    def _select_next_task(self) -> Optional[str]:
        """
        任务池调度：选择下一个要执行的任务
        
        硬约束：
        1. 已完成的任务不再重复
        2. recall_3words 必须在 registration_3words 完成后至少2分钟才能执行
        3. registration_3words 必须在 persona_collect 之后
        
        软约束（优先级）：
        1. 优先完成 persona_collect 收集用户信息
        2. orientation 任务优先（自然话题）
        3. registration 尽早执行以留出 recall 时间
        4. language 任务可穿插
        5. recall 在时间足够时执行
        6. attention_calculation 放在中后期
        """
        now = time.time()
        candidates = []
        
        # 🔍 诊断
        cognitive_done = [t for t in self.state._task_done if t not in self.catalog.buffer_tasks]
        self._log_verbose(f"任务选择前已完成: {cognitive_done}")
        
        for task_id in self.catalog.required_tasks:
            if task_id in self.catalog.visual_action_tasks:
                continue  # 需视觉/动作的任务暂不提问
            if task_id in self.state._task_done:
                continue

            until_turn = self.state._task_cooldown_until.get(task_id)
            if until_turn is not None and self.state._turn_counter < until_turn:
                continue
            
            cfg = self.catalog.task_config.get(task_id, {})
            
            # 硬约束：recall 必须等 registration 完成后 >= 2分钟
            if task_id == "recall_3words":
                if "registration_3words" not in self.state._task_done:
                    continue
                if self.state._registration_ts is None:
                    continue
                elapsed = now - self.state._registration_ts
                if elapsed < 120:  # 2分钟 = 120秒
                    self._log_verbose(f"recall_3words 需等待 {120 - elapsed:.0f}秒")
                    continue
            
            # 硬约束：registration 必须在 persona 收集后
            if task_id == "registration_3words":
                if "persona_collect_1" not in self.state._task_done or "persona_collect_2" not in self.state._task_done:
                    continue
            
            candidates.append(task_id)

        # 🔥 阶段门槛：基础闲聊+定向未完成前，不提前进入复杂指令/计算任务
        basic_prereq = {
            "persona_collect_1",
            "persona_collect_2",
            "orientation_time_year",
            "orientation_time_season",
            "orientation_time_month_date",
            "orientation_time_weekday",
            "orientation_place_province_city",
            "orientation_place_district",
            "orientation_place_location_floor",
        }
        advanced_tasks = {
            "attention_calc_life_math",
            "attention_reverse_phrase",
            "registration_3words",
            "recall_3words",
            "language_naming_watch",
            "language_naming_pencil",
            "language_repetition_sentence",
            "language_reading_close_eyes",
            "language_3step_action",
            "language_writing_sentence",
            "copy_pentagons",
        }
        if not basic_prereq.issubset(self.state._task_done):
            filtered = [t for t in candidates if t not in advanced_tasks]
            if filtered:
                if len(filtered) != len(candidates):
                    self._log_verbose("基础阶段：暂不进入复杂任务，优先闲聊/定向")
                candidates = filtered
        
        # 🔥 保存候选任务供异步分类使用
        self.state._available_candidates = candidates.copy()
        
        if not candidates:
            # 检查是否所有必需任务已完成
            remaining = set(self.catalog.required_tasks) - self.state._task_done
            if remaining:
                # 可能是 recall 还在等待
                if "recall_3words" in remaining and self.state._registration_ts:
                    elapsed = now - self.state._registration_ts
                    wait_time = 120 - elapsed
                    if wait_time > 0:
                        self._log_summary_card(
                            "Task Routing",
                            {"reason": "wait_recall", "wait_left": f"{wait_time:.0f}s", "next_task": "buffer_chat"},
                        )
                        return "buffer_chat"
            return None

        max_buffer_rounds = self._get_max_consecutive_buffer_chat()

        # 认知任务后插入 buffer_chat（但受计数器限制）
        if self.state._last_task_id and self._is_cognitive_task(self.state._last_task_id):
            if any(self._is_cognitive_task(t) for t in candidates):
                # 超过连续闲聊上限后，强制走任务选择
                if self.state._consecutive_buffer_count >= max_buffer_rounds:
                    self._log_summary_card(
                        "Task Routing",
                        {"reason": "buffer_limit", "buffer_count": self.state._consecutive_buffer_count},
                    )
                    # 继续走 LLM 选择逻辑
                else:
                    self.state._consecutive_buffer_count += 1
                    return "buffer_chat"
        
        # 🆕 用LLM动态选择下一个任务（根据对话上下文）
        if len(candidates) == 1:
            self.state._consecutive_buffer_count = 0  # 选到唯一任务，重置计数
            return candidates[0]
        
        # 🔥 优先检查摘要LLM的预建议（异步生成，0ms延迟）
        memory_tool = self._memory_tool_provider()
        if memory_tool is not None:
            sug = memory_tool.get_and_clear_suggestion()
            if sug:
                sug_task = sug.get('task_id', '')
                if sug_task in candidates:
                    from_topic = sug.get('from_topic', '')
                    to_topic = sug.get('to_topic', '')
                    anchor_fact = sug.get('anchor_fact', '')
                    bridge_hint = sug.get('bridge_hint', '')
                    target_question = sug.get('target_question', '')
                    self._set_bridge_context(
                        bridge_hint or (f"{from_topic}→{to_topic}" if from_topic else to_topic),
                        to_topic,
                        target_question=target_question,
                        target_task_id=sug_task,
                    )
                    self.state._current_turn_topic_set = True
                    self.state._consecutive_buffer_count = 0
                    self._log_summary_card(
                        "Task Routing",
                        {
                            "reason": "summary_suggestion",
                            "next_task": sug_task,
                            "anchor": anchor_fact or "无",
                            "bridge": self.state._last_bridge_hint,
                        },
                    )
                    return sug_task
                else:
                    self._log_verbose(f"摘要建议 '{sug_task}' 不在候选列表中，走LLM选择")
        
        return self._llm_select_task(candidates)
    def _normalize_text(self, text: str) -> str:
        s = (text or "").strip().lower()
        s = re.sub(r"\s+", "", s)
        s = re.sub(r"[\?？!！。．，,、:：;；\"'“”‘’\(\)（）\[\]【】{}]", "", s)
        return s
    def _normalize_topic_label(self, topic: str) -> str:
        return self._normalize_text(topic or "")
    def _categorize_topic(self, topic: str) -> str:
        """将话题文本映射到语义类别。返回类别名，未匹配返回原始文本。"""
        t = (topic or "").strip().lower()
        if not t:
            return ""
        for category, keywords in self.catalog.topic_categories.items():
            if any(kw in t for kw in keywords):
                return category
        return t  # 未匹配到任何类别，返回原始文本作为唯一类别
    def _is_recent_bridge_topic(self, topic: str, window: int = 99) -> bool:
        """检查目标话题是否与已用话题语义重复（全局检查，基于类别匹配）。"""
        cat = self._categorize_topic(topic)
        if not cat or not self.state._used_bridge_topics:
            return False
        # 精确匹配 + 语义类别匹配，双重检查
        norm = self._normalize_topic_label(topic)
        for used in self.state._used_bridge_topics:
            if self._normalize_topic_label(used) == norm:
                return True
            if self._categorize_topic(used) == cat and cat != topic.strip().lower():
                # 同类别视为重复（但排除未分类的原始文本直接比较）
                return True
        return False
    def _get_used_topic_categories(self) -> str:
        """返回已聊话题的去重类别列表，用于注入 LLM prompt。"""
        seen = []
        for t in self.state._used_bridge_topics:
            cat = self._categorize_topic(t)
            if cat and cat not in seen:
                seen.append(cat)
        return "、".join(seen) if seen else "无"
    def _remember_bridge_topic(self, topic: str) -> None:
        """记录本轮过渡目标话题（用于后续去重）。"""
        t = (topic or "").strip()
        if not t:
            return
        self.state._used_bridge_topics.append(t)
        cat = self._categorize_topic(t)
        self._log_verbose(f"记录话题: '{t}' → 类别: [{cat}] (已用类别: {self._get_used_topic_categories()})")
        if len(self.state._used_bridge_topics) > 20:
            self.state._used_bridge_topics = self.state._used_bridge_topics[-20:]
    def _set_bridge_context(
        self,
        hint: Optional[str],
        topic: Optional[str],
        remember_topic: bool = True,
        target_question: Optional[str] = None,
        target_task_id: Optional[str] = None,
    ) -> None:
        """统一设置自然语言过渡提示和结构化过渡主题。"""
        self.state._last_bridge_hint = (hint or "").strip() or None
        self.state._last_bridge_topic = (topic or "").strip() or None
        normalized_target_question = (target_question or "").strip() or None
        if target_task_id and normalized_target_question:
            aligned_target_question = self._normalize_target_question_candidate(target_task_id, normalized_target_question)
            if aligned_target_question and aligned_target_question != normalized_target_question:
                self._log_summary_card(
                    "Bridge Target Align",
                    {
                        "task": target_task_id,
                        "input": normalized_target_question,
                        "aligned": aligned_target_question,
                    },
                )
            normalized_target_question = aligned_target_question or normalized_target_question
        self.state._last_target_question = normalized_target_question
        self.state._last_target_task_id = target_task_id if normalized_target_question else None
        if remember_topic and self.state._last_bridge_topic:
            self._remember_bridge_topic(self.state._last_bridge_topic)
    def _is_similar_text(self, a: str, b: str) -> bool:
        na = self._normalize_text(a)
        nb = self._normalize_text(b)
        if not na or not nb:
            return False
        if na in nb or nb in na:
            return True
        # 使用 0.75 阈值，平衡防重复和多样性
        if SequenceMatcher(None, na, nb).ratio() >= 0.75:
            self._log_verbose(f"检测到相似问题 (ratio={SequenceMatcher(None, na, nb).ratio():.2f})")
            return True
        return False
    def _ensure_question_not_repeated(
        self, question: str, patient_profile: Dict, chat_history: List, task_id: Optional[str] = None
    ) -> str:
        if not question:
            return question
        return question
    def _extract_last_assistant_question(self, chat_history: List[Dict]) -> str:
        if not chat_history:
            return ""
        for msg in reversed(chat_history):
            if msg.get('role') == 'assistant':
                content = (msg.get('content') or '').strip()
                if content:
                    return content
        return ""
    def _question_focus(self, text: str) -> str:
        """提取问句核心，便于判断“同语义复问”"""
        import re
        t = (text or "").strip().replace("?", "？")
        if not t:
            return ""
        m = re.search(r'([^？]*？)\s*$', t)
        q = m.group(1) if m else t
        return self._normalize_text(q)
    def _question_semantic_slot(self, text: str) -> str:
        normalized = self._normalize_text(text)
        if not normalized:
            return ""
        slot_keywords = {
            "current_comfort": ["坐着", "舒服", "累不累", "撑得住", "缓一缓"],
            "meal": ["早饭", "午饭", "晚饭", "吃饭", "吃了没", "饿不饿"],
            "companion": ["陪着来", "谁陪", "家里人陪", "女儿陪", "儿子陪", "自己来", "一个人来"],
        }
        for slot, keywords in slot_keywords.items():
            if any(keyword in normalized for keyword in keywords):
                return slot
        return ""
    def _is_repetitive_buffer_question(self, new_question: str, chat_history: List[Dict]) -> bool:
        """检测 buffer 问题是否与上一轮问题语义过近。"""
        from difflib import SequenceMatcher

        prev_question = self._extract_last_assistant_question(chat_history)
        if not prev_question:
            return False

        if self._is_similar_text(new_question, prev_question):
            return True

        focus_new = self._question_focus(new_question)
        focus_prev = self._question_focus(prev_question)
        if focus_new and focus_prev:
            ratio = SequenceMatcher(None, focus_new, focus_prev).ratio()
            if ratio >= 0.62:
                self._log_verbose(f"检测到同语义复问 (focus_ratio={ratio:.2f})")
                return True
        slot_new = self._question_semantic_slot(new_question)
        slot_prev = self._question_semantic_slot(prev_question)
        if slot_new and slot_new == slot_prev:
            self._log_verbose(f"检测到同槽位复问 (slot={slot_new})")
            return True
        return False
    def _fallback_followup_question(self, patient_profile: Dict, chat_history: List) -> str:
        patient_name = patient_profile.get('name', '')
        greeting = f"{patient_name}，" if patient_name else ""
        last_user = ""
        for msg in reversed(chat_history or []):
            if msg.get('role') == 'user':
                last_user = (msg.get('content') or '').strip()
                break

        if any(token in last_user for token in ("省", "市", "区", "县", "医院")):
            return f"{greeting}地方我听明白了，咱们慢慢聊，您这会儿坐着还舒服不？"
        if any(token in last_user for token in ("堵车", "路上", "多久", "小时", "挂号", "检查")):
            return f"{greeting}咱们现在坐下来慢慢聊，您这会儿还舒服不？"
        if "综艺" in last_user:
            return f"{greeting}那您更喜欢轻松搞笑的，还是唱歌跳舞那种呀？"
        if "电视" in last_user or "节目" in last_user or "没看" in last_user or "没有" in last_user:
            return f"{greeting}那您平时更爱怎么解闷呀？听歌、刷手机，还是跟家里人聊聊天？"
        return f"{greeting}那您平时在家最喜欢干点啥？"
    def _is_cognitive_task(self, task_id: Optional[str]) -> bool:
        if not task_id:
            return False
        return task_id in self.catalog.task_config and task_id not in self.catalog.buffer_tasks
    def _needs_consent_for_task(self, task_id: str) -> bool:
        return False
    def _get_consent_group(self, task_id: Optional[str]) -> Optional[str]:
        if task_id in {"language_naming_watch", "language_naming_pencil"}:
            return "language_naming_visual"
        return None
    def _llm_select_task(self, candidates: List[str]) -> str:
        """Select a task through forced or topic-led routing strategies."""
        context = self._build_task_selection_context(candidates)
        if (
            self.state._consecutive_buffer_count >= context.max_buffer_rounds
            and context.non_buffer_candidates
        ):
            return self._select_forced_task(context)
        return self._select_topic_task(context)

    def _build_task_selection_context(
        self,
        candidates: List[str],
    ) -> TaskSelectionContext:
        non_buffer = [
            task for task in candidates if task != "buffer_chat"
        ]
        forced = list(non_buffer)
        if len(forced) > 1:
            blocked = {
                task
                for task in (
                    self.state._last_task_id,
                    self.state._last_forced_task_id,
                )
                if task in forced
            }
            if blocked and len(forced) > len(blocked):
                forced = [task for task in forced if task not in blocked]

        recent_lines = []
        for message in self.state.session_data.get("chat_history", [])[-4:]:
            content = message.get("content", "")[:50]
            if content:
                role = "对方" if message.get("role") == "user" else "你"
                recent_lines.append(f"{role}：{content}")
        recent_chat = "\n".join(recent_lines)

        memory_tool = self._memory_tool_provider()
        memory_topics = (
            memory_tool.get_discussed_topics_str_snapshot()
            if memory_tool is not None
            else ""
        )
        local_categories = (
            self._get_used_topic_categories()
            if self.state._used_bridge_topics
            else "无"
        )
        if memory_topics and memory_topics != "无":
            used_categories = memory_topics
            if local_categories != "无":
                used_categories += f"（补充：{local_categories}）"
        else:
            used_categories = local_categories
        self._log_verbose(f"已聊话题（防重复）: {used_categories}")
        return TaskSelectionContext(
            non_buffer_candidates=non_buffer,
            forced_candidates=forced,
            recent_chat=recent_chat,
            used_categories=used_categories,
            max_buffer_rounds=self._get_max_consecutive_buffer_chat(),
        )

    def _select_forced_task(
        self,
        context: TaskSelectionContext,
    ) -> str:
        self._log_summary_card(
            "Task Routing",
            {
                "reason": "forced_task_after_buffer_limit",
                "buffer_count": self.state._consecutive_buffer_count,
            },
        )
        candidates_text = ", ".join(
            f"{task}({TASK_SELECTION_DESCRIPTIONS.get(task, task)})"
            for task in context.forced_candidates
        )
        prompt = f"""你是认知筛查策略师。从以下任务中选一个，要能从当前话题自然过渡过去。
任务列表：{candidates_text}

最近对话：
{context.recent_chat or '（刚开始聊天）'}

【⚠️ 禁止重复】以下话题类别已经聊过，严禁再选类似话题：{context.used_categories}
选择原则：优先选能和当前话题衔接的任务。比如聊天气→聊日期/季节，聊做了啥→聊星期几，聊当前所在的地方→聊省/市/区县。
target_question 要写成下一轮最终真正要问用户的一句完整自然问句，只问 selected_task_id 这一件事，不能泄露答案；如果你拿不准，可留空字符串。
如果 selected_task_id 是 orientation_place_province_city / orientation_place_district / orientation_place_location_floor，target_question 必须问对方现在这个地方、当前所在的位置；禁止问老家、住址、常住地或平时住哪。
严格输出JSON：{{"from_topic":"当前话题","to_topic":"过渡话题","selected_task_id":"任务ID","target_question":"完整自然问句"}}"""
        try:
            llm = self._get_task_router_llm()
            data = self._parse_topic_json(
                self._invoke_task_router(llm, prompt)
            )
            selected = data.get("selected_task_id", "")
            from_topic = data.get("from_topic", "")
            to_topic = data.get("to_topic", "")
            target_question = data.get("target_question", "")
            if to_topic and self._is_recent_bridge_topic(to_topic, window=3):
                try:
                    retry_data = self._parse_topic_json(
                        self._invoke_task_router(
                            llm,
                            f"{prompt}\n注意：不要选「{to_topic}」，换一个不同话题。",
                        )
                    )
                    retry_topic = retry_data.get("to_topic", "")
                    if (
                        retry_topic
                        and not self._is_recent_bridge_topic(
                            retry_topic,
                            window=3,
                        )
                    ):
                        from_topic = retry_data.get("from_topic", "")
                        to_topic = retry_topic
                        selected = (
                            retry_data.get("selected_task_id") or selected
                        )
                        target_question = (
                            retry_data.get("target_question")
                            or target_question
                        )
                except Exception as retry_exc:
                    self._log_verbose(
                        f"强制模式重选失败: {retry_exc}"
                    )
            if selected in context.forced_candidates:
                self.state._consecutive_buffer_count = 0
                self._set_bridge_context(
                    (
                        f"{from_topic}→{to_topic}"
                        if from_topic
                        else to_topic
                    ),
                    to_topic,
                    target_question=target_question,
                    target_task_id=selected,
                )
                self.state._current_turn_topic_set = True
                self.state._last_forced_task_id = selected
                self._log_summary_card(
                    "Task Routing",
                    {
                        "reason": "forced_selection",
                        "next_task": selected,
                    },
                )
                return selected
        except Exception as exc:
            self._log_summary_card(
                "Task Routing",
                {
                    "reason": "forced_selection_failed",
                    "error": str(exc),
                },
            )
        self.state._consecutive_buffer_count = 0
        selected = random.choice(
            context.forced_candidates
            or context.non_buffer_candidates
        )
        self.state._last_forced_task_id = selected
        self._log_summary_card(
            "Task Routing",
            {
                "reason": "forced_random_fallback",
                "next_task": selected,
            },
        )
        return selected

    def _select_topic_task(
        self,
        context: TaskSelectionContext,
    ) -> str:
        prompt = f"""你是诊室聊天策略师。要通过自然聊天获得老人能回答的具体信息。
严格输出JSON：{{"from_topic":"当前话题","to_topic":"下一个话题","target_question":"如果已经能定下来，就给下一轮最终要问用户的完整自然问句；否则留空字符串"}}

【核心目标】选的话题要能自然引出以下可计分信息点之一：
- 时间感：今天星期几、几月几号、什么季节
- 地点感：当前这个地方在什么城市、什么区县
- 兴趣爱好：平时喜欢做什么
- 生活细节：今天吃了什么、做了什么
- 家人：家里人、孩子、老伴
- 当下状态：坐着舒不舒服、睡眠、简单活动
- 天气：今天天气怎么样
- 电视节目：最近看什么电视

【⚠️ 禁止重复】以下话题类别已经聊过，严禁再选：{context.used_categories}
必须选一个不同类别的新话题！

【策略】顺着用户说的话，过渡到上面的评估点。不要聊模糊的感受（如"心情""感觉"），要聊具体的事。
【target_question要求】如果你已经能把下一轮真正要问的句子定下来，就直接写成一句自然口语化问句；必须只问一件事，不能泄露答案。若当前只是粗话题还定不准，可输出空字符串。
示例：用户说"心情不错" → {{"from_topic":"心情","to_topic":"今天安排"}}（从心情好→问今天干什么具体事）
示例：用户说"在家看电视" → {{"from_topic":"看电视","to_topic":"星期几"}}（从看电视→今天周几有啥好节目）

最近对话：
{context.recent_chat or '（刚开始聊天）'}

输出JSON："""
        try:
            llm = self._get_task_router_llm()
            data = self._parse_topic_json(
                self._invoke_task_router(llm, prompt)
            )
            from_topic = data.get("from_topic", "")
            to_topic = data.get("to_topic", "")
            target_question = data.get("target_question", "")
            if to_topic and self._is_recent_bridge_topic(to_topic, window=3):
                try:
                    retry_data = self._parse_topic_json(
                        self._invoke_task_router(
                            llm,
                            f"{prompt}\n注意：不要选「{to_topic}」，换一个不同话题。",
                        )
                    )
                    retry_topic = retry_data.get("to_topic", "")
                    if (
                        retry_topic
                        and not self._is_recent_bridge_topic(
                            retry_topic,
                            window=3,
                        )
                    ):
                        from_topic = retry_data.get("from_topic", "")
                        to_topic = retry_topic
                        target_question = (
                            retry_data.get("target_question")
                            or target_question
                        )
                except Exception as retry_exc:
                    self._log_verbose(f"话题重选失败: {retry_exc}")
            bridge_hint = (
                f"{from_topic}→{to_topic}"
                if from_topic and to_topic
                else to_topic
            )
            direct_task = self._map_bridge_hint_to_task(
                bridge_hint,
                context.non_buffer_candidates,
            )
            self._set_bridge_context(
                bridge_hint,
                to_topic,
                target_question=target_question if direct_task else None,
                target_task_id=direct_task,
            )
            self.state._current_turn_topic_set = True
            if direct_task:
                self.state._consecutive_buffer_count = 0
                self.state._last_forced_task_id = direct_task
                return direct_task
            if (
                self.state._consecutive_buffer_count
                >= context.max_buffer_rounds
                and context.non_buffer_candidates
            ):
                return self._select_random_task(
                    context,
                    reason="buffer_limit_fallback",
                )
            self.state._consecutive_buffer_count += 1
            return "buffer_chat"
        except Exception as exc:
            self._log_summary_card(
                "Task Routing",
                {
                    "reason": "topic_selection_failed",
                    "error": str(exc),
                },
            )
            if (
                self.state._consecutive_buffer_count
                >= context.max_buffer_rounds
                and context.non_buffer_candidates
            ):
                return self._select_random_task(
                    context,
                    reason="topic_selection_random_fallback",
                )
            return "buffer_chat"

    def _select_random_task(
        self,
        context: TaskSelectionContext,
        *,
        reason: str,
    ) -> str:
        selected = random.choice(
            context.forced_candidates
            or context.non_buffer_candidates
        )
        self.state._consecutive_buffer_count = 0
        self.state._last_forced_task_id = selected
        self._log_summary_card(
            "Task Routing",
            {"reason": reason, "next_task": selected},
        )
        return selected

    def _get_task_router_llm(self):
        if self.use_local:
            from src.llm.model_pool import get_pooled_llm

            return get_pooled_llm(pool_key="7b_complex")
        from src.llm.http_client_pool import get_chat_openai

        return get_chat_openai(
            temperature=0.5,
            max_tokens=80,
            timeout=5,
            max_retries=1,
        )

    def _invoke_task_router(self, llm, prompt: str) -> str:
        started_at = time.time()
        max_retries = 3 if self.use_local else 2
        for retry in range(max_retries):
            try:
                response = llm.invoke(
                    [{"role": "user", "content": prompt}]
                )
                content = (
                    response.content
                    if hasattr(response, "content")
                    else str(response)
                )
                self._log_verbose(
                    f"任务路由耗时: {time.time() - started_at:.2f}s"
                )
                return content.strip()
            except Exception as exc:
                borrowed = "borrowed" in str(exc).lower()
                if self.use_local and borrowed and retry < max_retries - 1:
                    time.sleep(0.1 * (retry + 1))
                    continue
                raise
        raise RuntimeError("任务路由模型调用失败")

    def _parse_topic_json(self, content: str) -> Dict[str, str]:
        cleaned = re.sub(
            r"```json\s*|\s*```",
            "",
            (content or "").strip(),
        )
        data = json.loads(cleaned)
        return {
            "from_topic": (data.get("from_topic") or "").strip(),
            "to_topic": (data.get("to_topic") or "").strip(),
            "selected_task_id": (
                data.get("selected_task_id") or ""
            ).strip(),
            "target_question": (
                data.get("target_question") or ""
            ).strip(),
        }

    def _get_fixed_target_question(self, task_id: Optional[str]) -> Optional[str]:
        fixed_questions = {
            "orientation_time_year": "今年是哪一年来着？",
            "orientation_time_season": "您看现在是什么季节呀？",
            "orientation_time_month_date": "您记得今天几月几号吗？",
            "orientation_time_weekday": "今天星期几来着？",
            "orientation_place_province_city": "您现在这个医院是在什么省什么市呀？",
            "orientation_place_district": "您现在这个医院是在什么区或者县呀？",
            "orientation_place_location_floor": "您现在这个医院叫什么，在几楼呀？",
        }
        return fixed_questions.get(task_id)
    def _normalize_target_question_candidate(self, task_id: Optional[str], candidate_text: Optional[str]) -> Optional[str]:
        import re

        if not task_id or not candidate_text:
            return None

        candidate = (candidate_text or "").strip().strip('"').strip("'")
        if not candidate or "→" in candidate:
            return None

        location_task_ids = {
            "orientation_place_province_city",
            "orientation_place_district",
            "orientation_place_location_floor",
        }
        if task_id in location_task_ids:
            canonical = self._get_fixed_target_question(task_id)
            if canonical:
                return canonical

        normalized_candidate = self._normalize_text(candidate)

        if self.classify_question_sync(candidate, [task_id]) != task_id:
            return None

        question_like = (
            bool(re.search(r'[？?]', candidate))
            or bool(re.search(r'(想问|问下|您知道|你知道|您记得|你记得|来着|吗|呢)', candidate))
        )
        if not question_like:
            return None

        if task_id in location_task_ids:
            current_place_markers = [
                "现在", "这里", "这儿", "这个地方", "您这边", "你这边", "所在", "当前位置", "当前", "眼下", "这是",
            ]
            if not any(marker in normalized_candidate for marker in current_place_markers):
                canonical = self._get_fixed_target_question(task_id)
                if canonical:
                    return canonical

        candidate = candidate.rstrip('，,；;。！!~～ ')
        if not candidate.endswith(("？", "?")):
            candidate = f"{candidate}？"
        return candidate
    def _extract_target_question_core(self, task_id: Optional[str], candidate_text: Optional[str]) -> Optional[str]:
        import re

        surface = self._normalize_target_question_candidate(task_id, candidate_text)
        if not surface:
            return None

        candidate = surface.rstrip('，,；;。！!~～ ')
        sentence_parts = [seg.strip() for seg in re.split(r'[。！？!?；;]', candidate) if seg.strip()]
        if sentence_parts:
            candidate = sentence_parts[-1]

        comma_parts = [seg.strip() for seg in re.split(r'[，,]', candidate) if seg.strip()]
        if comma_parts:
            for seg in reversed(comma_parts):
                if re.search(r'[？?]|(想问|问下|您知道|你知道|您记得|你记得|来着|吗|呢)', seg):
                    candidate = seg
                    break
            else:
                candidate = comma_parts[-1]

        candidate = re.sub(r'^(那|那就|那您|那你|对了|顺便|再|然后|所以)\s*', '', candidate)
        candidate = re.sub(r'^(我想问(?:下)?您?|想问(?:下)?您?|问(?:下)?您?|请问您?|麻烦问您?)\s*', '', candidate)
        candidate = candidate.strip().strip('，,；;。！!~～ ')
        if not candidate:
            return surface

        if not candidate.endswith(("？", "?")):
            candidate = f"{candidate}？"

        if self.classify_question_sync(candidate, [task_id]) != task_id:
            return surface
        return candidate
    def _get_target_question_from_bridge_hint(self, task_id: Optional[str], bridge_hint: Optional[str]) -> Optional[str]:
        return self._normalize_target_question_candidate(task_id, bridge_hint)
    def _message_mentions_task_answer(self, task_id: Optional[str], text: Optional[str]) -> bool:
        import re

        message = (text or '').strip()
        if not task_id or not message:
            return False

        if task_id == "orientation_time_year":
            return bool(re.search(r'(?:19|20)\d{2}\s*年?', message))
        if task_id == "orientation_time_season":
            return any(token in message for token in ["春天", "夏天", "秋天", "冬天", "春季", "夏季", "秋季", "冬季"])
        if task_id == "orientation_time_month_date":
            return bool(re.search(r'\d{1,2}\s*月\s*\d{1,2}\s*[号日]?', message) or re.search(r'\d{1,2}\s*[号日]', message))
        if task_id == "orientation_time_weekday":
            return bool(re.search(r'(星期|周|礼拜)\s*[一二三四五六日天末1-7]', message))
        if task_id == "orientation_place_province_city":
            return bool(re.search(r'(省|市)', message))
        if task_id == "orientation_place_district":
            return bool(re.search(r'[\u4e00-\u9fa5]{1,12}(区|县)', message))
        if task_id == "orientation_place_location_floor":
            has_floor = bool(re.search(r'[一二三四五六七八九十百零\d]+\s*(楼|层)', message))
            has_place = any(token in message for token in ("医院", "门诊", "急诊", "病区", "科", "诊室", "大厅"))
            return has_floor or has_place
        return False
    def _was_task_answered_recently(self, task_id: Optional[str], conversation_history: Optional[List[Dict]]) -> bool:
        if not task_id or not conversation_history:
            return False

        recent_history = [msg for msg in conversation_history[-8:] if isinstance(msg, dict)]
        if not recent_history:
            return False

        canonical_question = self._get_fixed_target_question(task_id) or ""
        for idx in range(len(recent_history) - 1):
            assistant_msg = recent_history[idx]
            user_msg = recent_history[idx + 1]
            if assistant_msg.get('role') != 'assistant' or user_msg.get('role') != 'user':
                continue

            assistant_text = (assistant_msg.get('content') or '').strip()
            user_text = (user_msg.get('content') or '').strip()
            if not assistant_text or not user_text:
                continue

            asked_same_task = False
            try:
                asked_same_task = self.classify_question_sync(assistant_text, [task_id]) == task_id
            except Exception:
                asked_same_task = False

            if not asked_same_task and canonical_question:
                asked_same_task = self._is_similar_text(assistant_text, canonical_question)

            if asked_same_task and self._message_mentions_task_answer(task_id, user_text):
                return True

        return False
    def _latest_user_reply_mentions_target(self, task_id: Optional[str], conversation_history: Optional[List[Dict]]) -> bool:
        if not task_id or not conversation_history:
            return False

        last_user_msg = ""
        for msg in reversed(conversation_history):
            if msg.get('role') == 'user':
                last_user_msg = (msg.get('content') or '').strip()
                break

        if not last_user_msg:
            return False

        if self._message_mentions_task_answer(task_id, last_user_msg):
            return True
        return self._was_task_answered_recently(task_id, conversation_history)
