from __future__ import annotations

from collections.abc import Callable
from typing import List, Dict, Any, Optional, Protocol, Union, Generator, Set
import os
import json
import time
import re
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

from src.tools.agent_tools import (
    QueryTool,
    KnowledgeRetrievalTool,
    ResistanceDetectionTool,
    QuestionGenerationTool,
    ConversationStorageTool,
    AnswerEvaluationTool,
    ScoreRecordingTool,
    MMSEScoringTool,
    ComfortResponseTool,
    ImageDisplayTool,
    StandardQuestionTool,
    ConversationMemoryTool,
)
from src.agents.screening.catalog import ScreeningTaskCatalog
from src.agents.screening.state import ScreeningSessionState
from src.domain.dimensions import MMSE_DIMENSIONS
from src.utils.tool_logger import set_current_tool_log_session, log_summary


class TargetQuestionPolicy(Protocol):
    """Narrow planning policy required while preparing a tool request."""

    def _latest_user_reply_mentions_target(
        self,
        task_id: Optional[str],
        conversation_history: Optional[List[Dict]],
    ) -> bool: ...

    def _normalize_target_question_candidate(
        self,
        task_id: Optional[str],
        candidate_text: Optional[str],
    ) -> Optional[str]: ...

    def _extract_target_question_core(
        self,
        task_id: Optional[str],
        candidate_text: Optional[str],
    ) -> Optional[str]: ...

    def _get_target_question_from_bridge_hint(
        self,
        task_id: Optional[str],
        bridge_hint: Optional[str],
    ) -> Optional[str]: ...

    def _get_fixed_target_question(
        self,
        task_id: Optional[str],
    ) -> Optional[str]: ...


class ScreeningToolGateway:
    """Own external tools behind explicit state and policy dependencies."""

    def __init__(
        self,
        *,
        use_local: bool,
        state: ScreeningSessionState,
        catalog: ScreeningTaskCatalog,
        target_policy_provider: Callable[[], TargetQuestionPolicy],
        log_summary: Callable[[str, Dict[str, Any]], None],
        log_verbose: Callable[[str], None],
    ) -> None:
        self.use_local = use_local
        self.state = state
        self.catalog = catalog
        self._target_policy_provider = target_policy_provider
        self._log_summary_sink = log_summary
        self._log_verbose_sink = log_verbose
        # 常驻线程池 + 待消费的知识检索 future。检索是纯函数（除只读
        # current_dimension 外无 state 写入）且轮次串行，因此 future 可以
        # 安全跨阶段存活：Phase 1 提交、Phase 3 出题时才消费。
        # max_workers=2 而非 1：consume 超时放弃的慢检索会变成孤儿仍在后台
        # 跑（cancel 对已启动任务无效），若池只有 1 个 worker，下一轮的预取
        # 会排在孤儿后面被队头阻塞，预取效果反被架空。留一个余量 worker 吸收
        # 偶发孤儿——检索本就串行，正常情况下第二个 worker 空闲。
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="screening-prefetch",
        )
        self._pending_retrieval = None

    def start_retrieval_prefetch(
        self,
        dimension_name: str,
        conversation_history: List,
    ) -> None:
        """提交知识检索到常驻线程池，不阻塞当前阶段。"""
        self.cancel_retrieval_prefetch()
        query_result = self._call_query_generation(
            dimension_name,
            conversation_history,
        )
        self._pending_retrieval = self._prefetch_executor.submit(
            self._call_knowledge_retrieval,
            query_result,
        )
        self._log_verbose(f"知识检索预取已提交 (维度: {dimension_name})")

    def consume_retrieval_prefetch(
        self,
        *,
        dimension_name: str,
        conversation_history: List,
        timeout: float = 2.0,
    ) -> Dict[str, Any]:
        """取回预取的检索结果；未提交或超时则同步降级检索。"""
        pending = self._pending_retrieval
        self._pending_retrieval = None
        if pending is not None:
            try:
                return pending.result(timeout=timeout)
            except Exception as exc:
                self._log_verbose(f"预取检索取回失败，降级同步检索: {exc}")
        return self._call_knowledge_retrieval(
            self._call_query_generation(dimension_name, conversation_history)
        )

    def cancel_retrieval_prefetch(self) -> None:
        """丢弃上一轮遗留、未被消费的检索 future，避免跨轮陈旧结果。"""
        pending = self._pending_retrieval
        self._pending_retrieval = None
        if pending is not None:
            pending.cancel()

    def _latest_user_reply_mentions_target(self, *args, **kwargs):
        return self._target_policy_provider()._latest_user_reply_mentions_target(
            *args,
            **kwargs,
        )

    def _normalize_target_question_candidate(self, *args, **kwargs):
        return self._target_policy_provider()._normalize_target_question_candidate(
            *args,
            **kwargs,
        )

    def _extract_target_question_core(self, *args, **kwargs):
        return self._target_policy_provider()._extract_target_question_core(
            *args,
            **kwargs,
        )

    def _get_target_question_from_bridge_hint(self, *args, **kwargs):
        return self._target_policy_provider()._get_target_question_from_bridge_hint(
            *args,
            **kwargs,
        )

    def _get_fixed_target_question(self, *args, **kwargs):
        return self._target_policy_provider()._get_fixed_target_question(
            *args,
            **kwargs,
        )

    def _preview_log_value(self, value: Any, limit: int = 72) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return f"text_chars={len(value)}"
        if isinstance(value, (bytes, bytearray, memoryview)):
            return f"bytes={len(value)}"
        if isinstance(value, dict):
            return f"dict_fields={len(value)}"
        if isinstance(value, (list, tuple, set, frozenset)):
            return f"{type(value).__name__}_items={len(value)}"
        return str(value)
    def _log_summary_card(self, title: str, items: Dict[str, Any]) -> None:
        payload = {}
        for key, value in (items or {}).items():
            if value is None or value == "":
                continue
            payload[key] = self._preview_log_value(value)
        if payload:
            self._log_summary_sink(title, payload)
    def _log_verbose(self, message: str) -> None:
        self._log_verbose_sink(message)
    def _init_tools(self):
        """初始化所有工具"""
        self._log_verbose("初始化工具")

        # ⭐ 所有工具使用本地模式，避免网络延迟
        # 🔥 这些工具强制使用 API（可与本地模型并行执行）
        self.resistance_tool = ResistanceDetectionTool(use_local=False)
        self.comfort_tool = ComfortResponseTool(use_local=False)
        self.question_tool = QuestionGenerationTool(use_local=False)
        
        # 这些工具跟随 use_local 配置（需要本地 GPU）
        self.eval_tool = AnswerEvaluationTool(use_local=self.use_local)
        self.query_tool = QueryTool(use_local=self.use_local)
        
        # 其他工具
        self.score_tool = ScoreRecordingTool()  # 定性评估记录
        self.mmse_tool = MMSEScoringTool()  # ⭐ MMSE标准评分（30分制）
        self.retrieval_tool = KnowledgeRetrievalTool()
        self.storage_tool = ConversationStorageTool()
        self.image_tool = ImageDisplayTool()  # 📋 图片展示工具
        self.standard_question_tool = StandardQuestionTool(use_local=self.use_local)  # 📝 特殊维度问题工具
        self.memory_tool = ConversationMemoryTool(use_local=False)  # 🧠 对话记忆管理工具
        
        # ⭐ 追踪连续错误次数（用于 attention_calculation 提前终止）
        self.state.consecutive_failures = 0
        
        # ⭐ 连续减法状态跟踪（100-7 的当前值）
        self.state._calculation_current_value = 100  # 初始值
        self.state._calculation_step = 7  # 每次减去的数

        self._log_summary_card(
            "Tool Init",
            {
                "use_local": self.use_local,
                "api_parallel_tools": "resistance,comfort,question,memory",
                "follows_use_local": "eval,query,standard_question",
            },
        )
        
        # 预热工具
        if self.use_local:
            self._warmup_tools()
    def _warmup_tools(self):
        """预热关键工具"""
        self._log_verbose("预热工具")
        start_time = time.time()
        
        try:
            # 预热 Resistance + Eval（最常用）
            _ = self.resistance_tool._run(question="测试", answer="好的")
            _ = self.eval_tool._run(question="测试", answer="好的", task_id="orientation_time_weekday")
            
            warmup_time = time.time() - start_time
            self._log_summary_card("Tool Warmup", {"status": "ok", "elapsed": f"{warmup_time:.2f}s"})
        except Exception as e:
            self._log_summary_card("Tool Warmup", {"status": "failed", "error": str(e)})
    def _call_standard_question(
        self, 
        dimension_id: str, 
        is_dimension_switch: bool,
        memory_words: Optional[List[str]] = None,
        patient_name: Optional[str] = None,
        calculation_current_value: Optional[int] = None,
        calculation_step: Optional[int] = 7,
        last_user_message: Optional[str] = None,
    ) -> Dict:
        """
        调用特殊维度问题工具
        
        Args:
            dimension_id: 维度ID
            is_dimension_switch: 是否刚切换到此维度
            memory_words: 之前的记忆词（用于 recall 维度）
            patient_name: 患者姓名（用于个性化称呼）
            calculation_current_value: 连续减法的当前值（用于 attention_calculation）
            calculation_step: 连续减法的步长（默认7）
            last_user_message: 用户上一轮的话（用于自然衔接）
            
        Returns:
            包含 has_standard_question, question, memory_words 等字段的字典
        """
        result_json = self.standard_question_tool._run(
            dimension_id=dimension_id,
            is_dimension_switch=is_dimension_switch,
            memory_words=memory_words,
            patient_name=patient_name,
            calculation_current_value=calculation_current_value,
            calculation_step=calculation_step,
            last_user_message=last_user_message,
        )
        return json.loads(result_json)
    def _call_resistance_detection(self, question: str, answer: str) -> Dict:
        """调用抵抗检测工具"""
        with self._tool_log_scope():
            result_json = self.resistance_tool._run(question=question, answer=answer)
        return json.loads(result_json)
    def _call_comfort_response(
        self, resistance_result: Dict, patient_answer: str, patient_profile: Dict,
        chat_relaxed: bool = False  # 保留参数但不传递给工具
    ) -> Dict:
        """
        调用安慰话语生成工具
        
        Args:
            resistance_result: 抵抗检测结果
            patient_answer: 患者回答
            patient_profile: 患者画像
            chat_relaxed: 是否聊轻松话题（目前工具不使用此参数）
        """
        with self._tool_log_scope():
            result_json = self.comfort_tool._run(
                resistance_category=resistance_result.get('category', 'refusal'),
                patient_answer=patient_answer,
                resistance_reason=resistance_result.get('rationale'),
                patient_age=patient_profile.get('age'),
                patient_name=patient_profile.get('name'),
                patient_gender=patient_profile.get('gender'),
                used_topics=self.state._used_chat_topics,
                chat_history=self.state.session_data.get('chat_history')
            )
        
        # 更新已聊话题
        try:
            data = json.loads(result_json)
            if topic := data.get('selected_topic'):
                self.state._used_chat_topics.append(topic)
                self._log_verbose(f"记录新话题: {topic} (总计: {len(self.state._used_chat_topics)})")
        except:
            pass
            
        return json.loads(result_json)
    def _call_answer_evaluation(
        self,
        question: str,
        answer: str,
        task_id: str,
        patient_profile: Dict,
        expected_answer: Optional[str] = None,
    ) -> Dict:
        """调用回答评估工具"""
        result_json = self.eval_tool._run(
            question=question,
            answer=answer,
            task_id=task_id,
            expected_answer=expected_answer,
            patient_profile=patient_profile
        )
        return json.loads(result_json)
    def _call_query_generation(self, dimension_name: str, conversation_history: List) -> Dict:
        """规则化查询生成（替代 LLM 调用，省 1-3 秒）"""
        query = self.catalog.dimension_query_map.get(
            dimension_name,
            f"阿尔茨海默病 {dimension_name} 认知评估"
        )
        return {'query': query, 'keywords': query.split()}
    def _call_knowledge_retrieval(self, query_result: Dict) -> Dict:
        """调用知识检索工具"""
        # 知识检索单次约 1.9s 且串在首句 TTS 之前，而 MMSE 定向力/闲聊问题是固定
        # 问法、不依赖知识库，默认整体关闭；需要专业知识支撑时设 ENABLE_KNOWLEDGE_RETRIEVAL=true
        if os.getenv("ENABLE_KNOWLEDGE_RETRIEVAL", "false").strip().lower() not in {"1", "true", "yes", "on"}:
            return {'status': 'skipped', 'knowledge_context': '', 'results_count': 0}
        query = query_result.get('query', f"{self.state.current_dimension.get('name')} 认知评估")
        with self._tool_log_scope():
            result_json = self.retrieval_tool._run(query=query, top_k=3)
        return json.loads(result_json)
    def _call_question_generation(
        self, dimension_name: str, knowledge_context: str,
        patient_profile: Dict, conversation_history: List, is_followup: bool,
        is_dimension_switch: bool = False,
        needs_encouragement: bool = False,
        resistance_info: Dict = None,
        task_instruction: Optional[str] = None,
        persona_hooks: Optional[List[str]] = None,
        must_include: Optional[List[str]] = None,
        patient_emotion: str = 'neutral',
        task_id: Optional[str] = None,
        explicit_target_question: Optional[str] = None,
        explicit_target_question_core: Optional[str] = None,
    ) -> str:
        """调用问题生成工具"""
        # 提取患者信息
        patient_age = patient_profile.get('age')
        patient_education = patient_profile.get('education_years') or patient_profile.get('education')
        patient_name = patient_profile.get('name')
        patient_gender = patient_profile.get('gender', '女')
        
        # 🧠 使用 MemoryTool 获取上下文：摘要 + 近期窗口
        agent_state = {
            'task_done': self.state._task_done,
            'memory_words': self.state.session_data.get('memory_words'),
            'persona_hooks': persona_hooks,
        }
        mem_ctx = self.memory_tool.get_context(conversation_history or [], agent_state)
        history_info = mem_ctx['recent'] if mem_ctx['recent'] else None
        conversation_summary = mem_ctx['summary'] or None
        fixed_target_question = (explicit_target_question or '').strip() or None
        fixed_target_question_core = (explicit_target_question_core or '').strip() or None
        if fixed_target_question:
            if fixed_target_question_core:
                self._log_summary_card(
                    "Fixed Target",
                    {"task": task_id, "source": "explicit_target", "surface": fixed_target_question, "core": fixed_target_question_core},
                )
            else:
                self._log_summary_card(
                    "Fixed Target",
                    {"task": task_id, "source": "explicit_target", "surface": fixed_target_question},
                )
        elif task_id:
            if self._latest_user_reply_mentions_target(task_id, conversation_history):
                self._log_verbose(f"用户上一句已命中目标槽位，跳过固定问题路径: {task_id}")
            else:
                cached_target_question = None
                if self.state._last_target_task_id == task_id:
                    cached_target_question = self.state._last_target_question
                precomputed_target_question = self._normalize_target_question_candidate(
                    task_id,
                    cached_target_question,
                )
                if precomputed_target_question:
                    fixed_target_question = precomputed_target_question
                    fixed_target_question_core = self._extract_target_question_core(task_id, fixed_target_question)
                    if fixed_target_question_core and fixed_target_question_core != fixed_target_question:
                        self._log_summary_card(
                            "Fixed Target",
                            {"task": task_id, "source": "precomputed_target", "surface": fixed_target_question, "core": fixed_target_question_core},
                        )
                    else:
                        self._log_summary_card(
                            "Fixed Target",
                            {"task": task_id, "source": "precomputed_target", "surface": fixed_target_question},
                        )
                else:
                    bridge_target_question = self._get_target_question_from_bridge_hint(task_id, self.state._last_bridge_hint)
                    if bridge_target_question:
                        fixed_target_question = bridge_target_question
                        fixed_target_question_core = self._extract_target_question_core(task_id, fixed_target_question)
                        if fixed_target_question_core and fixed_target_question_core != fixed_target_question:
                            self._log_summary_card(
                                "Fixed Target",
                                {"task": task_id, "source": "bridge_hint", "surface": fixed_target_question, "core": fixed_target_question_core},
                            )
                        else:
                            self._log_summary_card(
                                "Fixed Target",
                                {"task": task_id, "source": "bridge_hint", "surface": fixed_target_question},
                            )
                    else:
                        canonical_target_question = self._get_fixed_target_question(task_id)
                        if canonical_target_question:
                            fixed_target_question = canonical_target_question
                            fixed_target_question_core = self._extract_target_question_core(task_id, fixed_target_question)
                            self._log_summary_card(
                                "Fixed Target",
                                {"task": task_id, "source": "canonical_fallback", "surface": fixed_target_question},
                            )

        # 🔥 流式回调：如果 agent 有 _stream_sentence_cb，注入到 question_tool
        stream_cb = self.state._stream_sentence_cb
        if stream_cb:
            self.question_tool._on_sentence_cb = stream_cb

        try:
            with self._tool_log_scope():
                result_json = self.question_tool._run(
                    dimension_name=dimension_name,
                    dimension_description="",
                    knowledge_context=knowledge_context,
                    patient_age=patient_age,
                    patient_education=patient_education,
                    patient_name=patient_name,
                    patient_gender=patient_gender,
                    conversation_history=history_info,
                    conversation_summary=conversation_summary,
                    patient_emotion=patient_emotion,
                    task_instruction=task_instruction,
                    task_id=task_id,
                    persona_hooks=persona_hooks,
                    must_include=must_include,
                    avoid_questions=self.state._asked_questions,
                    bridge_hint=self.state._last_bridge_hint,
                    target_question=fixed_target_question,
                    target_question_core=fixed_target_question_core,
                )
        finally:
            # 清理回调，避免影响其他路径
            self.question_tool._on_sentence_cb = None

        result = json.loads(result_json)
        return result.get('question', '请继续')
    @contextmanager
    def _tool_log_scope(self):
        set_current_tool_log_session(self.state.session_id)
        try:
            yield
        finally:
            set_current_tool_log_session(None)
    def _check_and_display_image(self, question: str, session_id: str) -> Dict:
        """
        检测问题是否需要展示图片
        
        根据问题内容判断是否是命名/阅读任务，并返回相应的图片展示指令
        """
        question_lower = question.lower()
        
        # 检测关键词
        naming_keywords = ['这是什么', '叫什么', '名字', '物品', '东西']
        reading_keywords = ['闭上眼睛', '照着做', '按照', '文字', '看到']
        
        # 判断是否是命名任务
        if any(keyword in question for keyword in naming_keywords):
            # 随机选择手表或铅笔（或者根据评估次数轮换）
            # 这里简化处理，根据session_id哈希决定
            import hashlib
            hash_val = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
            image_id = 'watch' if hash_val % 2 == 0 else 'pencil'
            title = "请看下面的图片，这是什么东西？"
            
            result_json = self.image_tool._run(
                image_id=image_id,
                title=title,
                action='show'
            )
            result = json.loads(result_json)
            
            return {
                'should_display': result.get('success', False),
                'image_id': image_id,
                'display_command': result.get('display_command')
            }
        
        # 判断是否是阅读任务
        elif any(keyword in question for keyword in reading_keywords):
            title = "请照着上面的文字做"
            
            result_json = self.image_tool._run(
                image_id='close_eyes',
                title=title,
                action='show'
            )
            result = json.loads(result_json)
            
            return {
                'should_display': result.get('success', False),
                'image_id': 'close_eyes',
                'display_command': result.get('display_command')
            }
        
        # 不需要展示图片
        return {
            'should_display': False,
            'image_id': None,
            'display_command': None
        }
    def _call_conversation_storage(self, session_id: str, user_input: str, generated_question: str):
        """调用对话存储工具"""
        # 构建 turn_data
        turn_data = {
            "user_question": user_input,
            "assistant_response": generated_question,
            "dimension_id": self.state.current_dimension.get('id'),
            "dimension_name": self.state.current_dimension.get('name')
        }
        
        self.storage_tool._run(
            session_id=session_id,
            action='save_turn',
            turn_data=json.dumps(turn_data, ensure_ascii=False)
        )
    def _call_natural_transition(
        self, user_answer: str, dimension_name: str, 
        patient_profile: Dict, chat_history: List, current_emotion: str
    ) -> str:
        """调用工具生成自然过渡回应"""
        result_json = self.question_tool.generate_natural_transition(
            user_answer=user_answer,
            dimension_name=dimension_name,
            patient_name=patient_profile.get('name'),
            patient_gender=patient_profile.get('gender'),
            patient_age=patient_profile.get('age'),
            chat_history=chat_history,
            current_emotion=current_emotion,
        )
        result = json.loads(result_json)
        return result.get('transition', '嗯，您说得对。')
