from __future__ import annotations

from collections.abc import Callable
from typing import List, Dict, Any, Optional, Union, Generator, Set
import os
import json
import time
import re
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

from src.agents.screening.catalog import ScreeningTaskCatalog
from src.agents.screening.state import ScreeningSessionState
from src.agents.screening.tool_gateway import ScreeningToolGateway
from src.domain.dimensions import MMSE_DIMENSIONS
from src.utils.tool_logger import set_current_tool_log_session, log_summary


class ScreeningBackgroundAnalysis:
    """Analyze topic/history snapshots using explicit state and tool dependencies."""

    def __init__(
        self,
        *,
        state: ScreeningSessionState,
        catalog: ScreeningTaskCatalog,
        tools: ScreeningToolGateway,
        log_summary: Callable[[str, Dict[str, Any]], None],
        log_verbose: Callable[[str], None],
        llm_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.state = state
        self.catalog = catalog
        self.tools = tools
        self._log_summary_sink = log_summary
        self._log_verbose_sink = log_verbose
        self._llm_factory = llm_factory or self._default_llm_factory

    @staticmethod
    def _default_llm_factory(model: str):
        from src.llm.http_client_pool import get_siliconflow_chat_openai

        return get_siliconflow_chat_openai(
            model=model,
            temperature=0.3,
            timeout=10,
            max_retries=1,
        )

    def _log_summary_card(self, title: str, items: Dict[str, Any]) -> None:
        self._log_summary_sink(title, items)

    def _log_verbose(self, message: str) -> None:
        self._log_verbose_sink(message)

    async def analyze(
        self,
        topic: str,
        chat_history: List[Dict],
    ) -> None:
        """Run independent background topic and history analyses."""
        session_id = self.state.session_id
        self._log_verbose(f"后台侦探启动：Topic='{topic}'")
        await self._map_background_topic(topic, session_id)
        await self._scan_background_history(chat_history, session_id)

    def _session_is_current(self, session_id: Optional[str]) -> bool:
        return self.state.session_id == session_id

    async def _map_background_topic(
        self,
        topic: str,
        session_id: Optional[str],
    ) -> None:
        """Map a transition topic to one safe lightweight task."""
        if topic:
            try:
                undone_tasks = [t for t in self.catalog.required_tasks if t not in self.state._task_done]
                self._log_verbose(f"后台待完成任务列表: {undone_tasks}")
                
                if not undone_tasks:
                    self._log_verbose("所有任务已完成，跳过 topic 映射")
                else:
                    # 从 topic 中提取（topic 格式可能是 "吃饭→算术" 或 "算术"）
                    topic_text = topic.split("→")[-1].strip() if "→" in topic else topic.strip()

                    # 仅允许将话题映射到轻量任务，避免直接跳到复杂动作/计算任务
                    topic_mappable = {
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
                    mappable_tasks = [t for t in undone_tasks if t in topic_mappable]
                    if not mappable_tasks:
                        self._log_verbose("当前无可话题直达的轻量任务，跳过 topic 映射")
                        mappable_tasks = []

                    # 全量使用 LLM 判断话题映射，不走关键词优先
                    if mappable_tasks:
                        background_model = os.getenv(
                            "BACKGROUND_ANALYSIS_MODEL",
                            os.getenv("TASK_ROUTER_MODEL", "Qwen/Qwen2.5-7B-Instruct")
                        )
                        llm = self._llm_factory(background_model)
                        # 🔁 将已使用过的过渡话题显式告诉 LLM，降低再次选中的概率
                        recent_topics = [t for t in self.state._used_bridge_topics[-6:] if t]
                        recent_topics_text = "、".join(recent_topics) if recent_topics else "（暂无）"

                        prompt_topic = f"""当前对话正过渡到话题：「{topic_text}」。
请判断这个话题是否直接对应以下待完成任务之一：
{', '.join(mappable_tasks)}

如果对应，输出任务ID。
否则输出 None。
只输出结果，不要解释。

【话题使用记录】本次会话里，之前已经用过这些话题做过渡：{recent_topics_text}
如果当前话题和这些话题高度相似（比如都是围绕“星期几”“日期”反复提问），
而且之前已经通过这些话题成功完成了相关任务，请**优先输出 None**，避免来回重复同一个话题。"""
                        res = await llm.ainvoke([{"role": "user", "content": prompt_topic}])
                        res_content = res.content.strip()

                        if not self._session_is_current(session_id):
                            self._log_verbose(
                                "后台话题映射对应的会话已切换，丢弃旧结果"
                            )
                            return

                        if res_content in mappable_tasks:
                            # 🛡️ 防御：仅在 _precomputed_next_task 为空时写入，避免覆盖 process_turn 的选择
                            if self.state._precomputed_next_task is not None:
                                self._log_verbose(f"_precomputed_next_task 已有值 '{self.state._precomputed_next_task}'，跳过覆盖")
                            elif res_content in self.state._task_done:
                                self._log_verbose(f"Topic命中 '{res_content}' 但已在 _task_done 中，跳过")
                            elif res_content == self.state._last_task_id:
                                self._log_verbose(f"Topic命中 '{res_content}' 但这是当前活动任务，跳过")
                            else:
                                self._log_summary_card(
                                    "Background Detect",
                                    {"reason": "topic_match", "topic": topic_text, "task": res_content},
                                )
                                self.state._precomputed_next_task = res_content
                                self.state._consecutive_buffer_count = 0  # 重置计数器
                        else:
                            self._log_verbose(f"Topic未命中任何任务: '{topic_text}' -> '{res_content}'")
                            
            except Exception as e:
                self._log_summary_card("Background Detect", {"reason": "topic_mapping_failed", "error": str(e)})

    async def _scan_background_history(
        self,
        chat_history: List[Dict],
        session_id: Optional[str],
    ) -> None:
        """Detect already answered lightweight tasks in recent history."""
        _GLOBAL_SCAN_ALLOWED = {
            "orientation_time_year", "orientation_time_season",
            "orientation_time_month_date", "orientation_time_weekday",
            "orientation_place_province_city", "orientation_place_district",
            "orientation_place_location_floor",
            "persona_collect_1", "persona_collect_2",
        }
        try:
            # 🛡️ 排除当前正在等待用户回答的任务（AI刚问了，用户还没答）
            _active_task = self.state._last_task_id
            undone = [t for t in self.catalog.required_tasks
                      if t not in self.state._task_done and t in _GLOBAL_SCAN_ALLOWED and t != _active_task]
            if not undone:
                return

            # 构建最近历史文本
            history_text = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in chat_history[-10:]]) # 限制长度
            
            prompt_global = f"""请审查以下对话历史，判断用户是否**已经回答**了以下任务的问题：
待完成任务：{', '.join(undone)}

任务说明：
- orientation_time_weekday: 说了星期几
- orientation_place_city: 说了所在城市
- persona_collect: 说了兴趣或习惯

输出格式：JSON list，如 ["task_id_1", "task_id_2"]。如果无，输出 []。

对话历史：
{history_text}
"""
            background_model = os.getenv(
                "BACKGROUND_ANALYSIS_MODEL",
                os.getenv("TASK_ROUTER_MODEL", "Qwen/Qwen2.5-7B-Instruct")
            )
            global_llm = self._llm_factory(background_model)
            res = await global_llm.ainvoke([{"role": "user", "content": prompt_global}])
            content = res.content.strip()

            if not self._session_is_current(session_id):
                self._log_verbose(
                    "后台历史扫描对应的会话已切换，丢弃旧结果"
                )
                return
            
            import re
            import json
            content = re.sub(r'```json\s*|\s*```', '', content)
            
            # 🔥 改进的JSON解析：尝试提取JSON数组
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                hit_tasks = json.loads(json_match.group())
            elif content.strip() == '[]' or content.strip().lower() == 'none' or not content.strip():
                hit_tasks = []
            else:
                # 尝试直接解析（兼容旧格式）
                hit_tasks = json.loads(content)
            
            if isinstance(hit_tasks, list) and hit_tasks:
                # 限制每轮最多自动完成 1 个任务，避免误判导致进度跳跃
                try:
                    max_auto_done = int(os.getenv("GLOBAL_SCAN_MAX_AUTO_DONE", "1"))
                except ValueError:
                    max_auto_done = 1
                max_auto_done = max(0, max_auto_done)

                unique_hits = []
                for task in hit_tasks:
                    if task in undone and task not in unique_hits:
                        unique_hits.append(task)

                capped_hits = unique_hits[:max_auto_done] if max_auto_done > 0 else []
                if capped_hits:
                    self._log_summary_card(
                        "Background Detect",
                        {
                            "reason": "global_scan_hit",
                            "hits": unique_hits,
                            "applied": capped_hits,
                        },
                    )
                    if len(unique_hits) > len(capped_hits):
                        self._log_verbose(
                            f"本轮仅自动完成 {len(capped_hits)} 个任务，其余延后确认: {unique_hits[len(capped_hits):]}"
                        )
                else:
                    self._log_verbose("全局扫描无可自动确认任务")

                for done_task in capped_hits:
                    if done_task in undone:
                        # 标记任务完成
                        self.state._task_done.add(done_task)
                        self._log_summary_card("Background Detect", {"reason": "auto_complete", "task": done_task})
                        
                        # 重要：同时记录一个满分评分，防止空缺
                        try:
                            # 尝试找到对应的维度ID
                            task_cfg = self.catalog.task_config.get(done_task, {})
                            dim_id = task_cfg.get('dimension_id')
                            max_points = task_cfg.get('max_points', 1)
                            
                            if dim_id:
                                # 1. 异步记录定性评分
                                self.tools.score_tool._run(
                                    session_id=session_id or 'unknown',
                                    dimension_id=dim_id,
                                    quality_level="good",
                                    cognitive_performance="隐式回答正确",
                                    question="(后台全局扫描)",
                                    answer="(历史对话隐式包含)",
                                    evaluation_detail="全局历史扫描发现用户已回答此问题",
                                    action="save"
                                )
                                
                                # 2. 🔥 必须调用 MMSE 定量评分，否则报告里是0分
                                mmse_res = self.tools.mmse_tool._run(
                                    session_id=session_id or 'unknown',
                                    dimension_id=dim_id,
                                    score=max_points,
                                    max_score=max_points,
                                    task_id=done_task,
                                    question="(后台全局扫描)",
                                    answer="(历史对话隐式包含)",
                                    evaluation_detail=f"后台自动补记分: {done_task}",
                                    action="save"
                                )
                                self._log_verbose(f"MMSE自动记分: {done_task} (+{max_points}) -> {mmse_res}")
                                
                        except Exception as e_score:
                             self._log_summary_card("Background Detect", {"reason": "auto_score_failed", "task": done_task, "error": str(e_score)})
            else:
                self._log_verbose("全局扫描无新发现")
            
        except Exception as e:
             self._log_summary_card("Background Detect", {"reason": "global_scan_failed", "error": str(e)})
