from __future__ import annotations

from collections.abc import Callable
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from .answer_evaluation import ScreeningAnswerEvaluation
from .catalog import ScreeningTaskCatalog
from .conversation_policy import ScreeningConversationPolicy
from .question_classifier import ScreeningQuestionClassifier
from .state import ScreeningSessionState
from .task_planning import ScreeningTaskPlanning
from .tool_gateway import ScreeningToolGateway
from .turn_models import (
    AnswerPathResult,
    EvaluationPhaseResult,
    ParallelResistanceResult,
)


class ScreeningAnswerPhase:
    """Evaluate the answer to the question issued by the previous turn."""

    def __init__(
        self,
        *,
        state: ScreeningSessionState,
        catalog: ScreeningTaskCatalog,
        question_classifier: ScreeningQuestionClassifier,
        task_planner: ScreeningTaskPlanning,
        tools: ScreeningToolGateway,
        policy: ScreeningConversationPolicy,
        answer_evaluator: ScreeningAnswerEvaluation,
        log_summary: Callable[[str, Dict[str, Any]], None],
        log_verbose: Callable[[str], None],
    ) -> None:
        self.state = state
        self.catalog = catalog
        self.question_classifier = question_classifier
        self.task_planner = task_planner
        self.tool_gateway = tools
        self.policy = policy
        self.answer_evaluator = answer_evaluator
        self._log_summary = log_summary
        self._log_verbose_sink = log_verbose

    def _log_summary_card(
        self,
        title: str,
        items: Dict[str, Any],
    ) -> None:
        self._log_summary(title, items)

    def _log_verbose(self, message: str) -> None:
        self._log_verbose_sink(message)

    def evaluate_previous_turn(
        self,
        *,
        user_answer: str,
        doctor_question: str,
        patient_profile: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        dimension_id: str,
        dimension_name: str,
        start_time: float,
    ) -> EvaluationPhaseResult | Dict[str, Any]:
        """Evaluate the answer to the previously issued task."""
        expected_answer: Optional[str] = None
        # ==================== 评估模式：正常流程 ====================
        step1_start = time.time()
        
        # 🔥 标记本轮是否运行了 Step 1（用于后续判断话题是否新鲜）
        self.state._current_turn_topic_set = False
        
        # 🔥 获取后台分类结果（仅供参考，不覆盖 last_task_id）
        # 分类结果仅用于诊断日志，绝不阻塞主链路：非阻塞查询，
        # 未就绪就退回上次结果（每轮省掉最坏 0.5s 的白等）。
        classification_result = self.question_classifier.result(timeout=0)
        
        # 🆕 任务池：获取上一个任务的信息（这是实际执行的任务）
        last_task_id = self.state._last_task_id
        # 🔍 诊断：每轮开始打印状态
        _diag_cognitive = [t for t in self.state._task_done if t not in self.catalog.buffer_tasks]
        self._log_verbose(
            f"轮开始 | last_task={last_task_id} | done={_diag_cognitive} | turn={self.state._turn_counter}"
        )
        
        # 🔥 修复：分类结果只在特定场景使用（如评估任务时确认问题类型）
        # 不应该覆盖 last_task_id，因为 last_task_id 代表"实际执行的任务"
        # 而 classification_result 代表"AI上一轮提问的类型"
        if classification_result and classification_result != "buffer_chat":
            if last_task_id != classification_result:
                self._log_verbose(f"分类参考: {classification_result} (实际任务: {last_task_id})")
        
        last_task_is_buffer = (last_task_id in self.catalog.buffer_tasks) if last_task_id else True
        last_task_cfg = self.catalog.task_config.get(last_task_id, {}) if last_task_id else {}
        task_dimension_id = last_task_cfg.get('dimension_id') or dimension_id
        # 所有评估快速路径都会在后续读取该标记；只有并行抵抗检测路径
        # 会重新赋值，必须先给规则评估/短回答路径一个稳定默认值。
        invalid_skip_from_resistance = False
        
        # 如果上一轮是缓冲任务，跳过评分但仍检测抵抗
        if last_task_is_buffer:
            answer_path = self._evaluate_buffer_answer(
                last_task_id=last_task_id,
                user_answer=user_answer,
                doctor_question=doctor_question,
                patient_profile=patient_profile,
                chat_history=chat_history,
                dimension_id=dimension_id,
                dimension_name=dimension_name,
                start_time=start_time,
            )
        else:
            answer_path = self._evaluate_cognitive_answer(
                last_task_id=last_task_id,
                last_task_cfg=last_task_cfg,
                user_answer=user_answer,
                doctor_question=doctor_question,
                patient_profile=patient_profile,
                chat_history=chat_history,
                dimension_id=dimension_id,
                dimension_name=dimension_name,
                start_time=start_time,
                step1_start=step1_start,
            )
        if isinstance(answer_path, dict):
            return answer_path
        expected_answer = answer_path.expected_answer
        eval_result = answer_path.evaluation
        resistance_result = answer_path.resistance

        step1_time = time.time() - step1_start
        self._log_summary_card(
            "Phase 1",
            {
                "Path": 'buffer' if last_task_is_buffer else 'task',
                "Resistance": resistance_result.get('category') or '-',
                "Intent": resistance_result.get('turn_intent') or '-',
                "Time": f"{step1_time:.2f}s",
            },
        )
    
        return EvaluationPhaseResult(
            last_task_id=last_task_id,
            expected_answer=expected_answer,
            evaluation=eval_result,
            elapsed=step1_time,
        )

    def _evaluate_buffer_answer(
        self,
        *,
        last_task_id: Optional[str],
        user_answer: str,
        doctor_question: str,
        patient_profile: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        dimension_id: str,
        dimension_name: str,
        start_time: float,
    ) -> AnswerPathResult | Dict[str, Any]:
        """Evaluate a non-scoring buffer answer and prefetch routing work."""
        self._log_verbose(f"上轮是缓冲任务({last_task_id})，跳过评分但检测抵抗")
        
        # 🔥 v1.2: 快速抵抗预判 — 短/积极回答直接跳过 ResistanceTool (省 2-4s)
        _positive_kw = ['好', '是', '对', '嗯', '行', '可以', '不错', '舒畅', '开心', '高兴', '挺好']
        _negative_kw = ['不', '烦', '累', '别', '走', '算了', '不想', '讨厌', '没意思', '无聊']
        _answer_stripped = user_answer.strip()
        _is_clearly_positive = (
            len(_answer_stripped) <= 20
            and any(kw in _answer_stripped for kw in _positive_kw)
            and not any(kw in _answer_stripped for kw in _negative_kw)
        )
        
        if _is_clearly_positive:
            # ⚡ 极速路径：明显积极回答，跳过抵抗检测 API
            self._log_verbose(f"buffer极速路径: 积极回答'{_answer_stripped[:15]}'，跳过抵抗检测")
            resistance_result = {
                'is_resistant': False,
                'confidence': 1.0,
                'category': 'none',
                'wants_repeat': False,
                'has_substantive_answer': bool(_answer_stripped),
                'turn_intent': 'answer',
            }
            
            # 知识检索交给常驻线程池预取（Phase 3 出题时消费），本地线程池
            # 只跑必须在本阶段拿到结果的任务选择。
            self.tool_gateway.start_retrieval_prefetch(dimension_name, chat_history)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future_task_select = None
                if not self.state._pending_consent_task_id:
                    future_task_select = executor.submit(
                        self.task_planner._select_next_task
                    )
                self._log_verbose("buffer极速路径: 任务选择执行 + 知识检索预取")
        else:
            # 正常路径：并行执行抵抗检测 + 任务选择 + 知识预取
            self._log_verbose("buffer路径: 并行执行抵抗检测(API)+任务选择+知识预取")
        
            # 知识检索交给常驻线程池预取（Phase 3 出题时消费）。
            self.tool_gateway.start_retrieval_prefetch(dimension_name, chat_history)
            with ThreadPoolExecutor(max_workers=2) as executor:
                # 1. 抵抗检测
                future_resist = executor.submit(
                    self.tool_gateway._call_resistance_detection,
                    doctor_question,
                    user_answer,
                )

                # 2. 提前启动任务选择
                future_task_select = None
                if not self.state._pending_consent_task_id:
                    future_task_select = executor.submit(
                        self.task_planner._select_next_task
                    )
                self._log_verbose(f"buffer路径: 抵抗检测+任务选择执行 + 知识检索预取 (维度: {dimension_name})")

                # 获取抵抗检测结果
                resistance_result = future_resist.result()
                invalid_skip_from_resistance = False
        
        # 🔥 处理请求重复：只有明确要求重复且未给出实质回答时，才直接重复上一个问题
        if resistance_result.get('wants_repeat') and not resistance_result.get('has_substantive_answer'):
            self._log_summary_card(
                "Repeat Request",
                {
                    "action": "repeat_previous_question",
                    "dimension": dimension_name,
                    "task": self.state._last_task_id,
                },
            )
            repeat_prefixes = [
                "好的，我再说一遍：",
                "没关系，我再问一次：",
                "好啊，我重复一下：",
                "您说的对，我再说一次：",
            ]
            import random
            prefix = random.choice(repeat_prefixes)
            repeated_question = f"{prefix}{doctor_question}"
        
            return {
                'output': repeated_question,
                'response': repeated_question,
                'is_repeat': True,
                'dimension': dimension_name,
                'dimension_id': dimension_id,
                'total_time': time.time() - start_time
            }
        if resistance_result.get('category') == 'repeat_request' and resistance_result.get('has_substantive_answer'):
            self._log_verbose("检测到重复倾向，但用户已给出实质回答，不重复上一问题")
    
        # 🔥 修复：buffer 任务检测到抵抗也需要安慰 -> 直接中断返回
        if resistance_result.get('is_resistant'):
            category = resistance_result.get('category', 'unknown')
            self._log_summary_card(
                "Resistance Interrupt",
                {"path": "buffer", "category": category, "action": "enter_comfort_mode"},
            )
            self.state.is_in_comfort_mode = True
            self.state.comfort_turn_count = 0
            self.state._comfort_entry_category = category
            interrupted_task = self.policy._get_recent_cognitive_task_id(last_task_id)
            if interrupted_task and interrupted_task not in self.catalog.buffer_tasks:
                self.state._comfort_interrupted_task_id = interrupted_task
        
            # 调用 comfort 工具生成安抚话术
            comfort_result = self.tool_gateway.comfort_tool._run(
                resistance_category=category,
                patient_answer=user_answer,
                patient_name=patient_profile.get('name'),
                patient_age=patient_profile.get('age'),
                patient_gender=patient_profile.get('gender'),
                used_topics=self.state._used_chat_topics,
                chat_history=chat_history,
                use_template=False,
            )
        
            # 解析结果
            comfort_data = json.loads(comfort_result)
            if topic := comfort_data.get('selected_topic'):
                self.state._used_chat_topics.append(topic)
                self.task_planner._set_bridge_context(
                    topic,
                    topic,
                    remember_topic=False,
                )
        
            comfort_message = comfort_data.get('comfort_message', '您说得对，咱们继续聊聊。')
            self.state._last_generated_question = comfort_message
        
            return {
                'output': comfort_message,
                'response': comfort_message,
                'has_resistance': True,
                'resistance_category': category,
                'is_comfort_mode': True,
                    'dimension': dimension_name,
                'dimension_id': 'buffer',
                'total_time': time.time() - start_time
            }

        # 🔥 无抵抗：知识检索已作为常驻预取提交，Phase 3 出题时消费。

        # 🔥 收集任务选择结果（关键！Phase 2 直接用，省掉 ~2s LLM 调用）
        if future_task_select:
            self.state._precomputed_next_task = future_task_select.result()
            topic_info = self.state._last_bridge_hint or "N/A"
            self._log_summary_card(
                "Buffer Pre-Selection",
                {
                    "Task": self.state._precomputed_next_task,
                    "Bridge": topic_info,
                    "Intent": resistance_result.get('turn_intent') or '-',
                },
            )
    
        eval_result = {
            'is_correct': True, 'quality_level': 'good', 'cognitive_performance': '正常',
            'is_complete': True, 'evaluation_detail': '缓冲闲聊轮不计分',
            'need_followup': False, 'confidence': 1.0
        }
        return AnswerPathResult(
            expected_answer=None,
            evaluation=eval_result,
            resistance=resistance_result,
        )

    def _evaluate_cognitive_answer(
        self,
        *,
        last_task_id: str,
        last_task_cfg: Dict[str, Any],
        user_answer: str,
        doctor_question: str,
        patient_profile: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        dimension_id: str,
        dimension_name: str,
        start_time: float,
        step1_start: float,
    ) -> AnswerPathResult | Dict[str, Any]:
        """Evaluate a scoring task answer using fast or parallel paths."""
        # 获取期望答案（用于精准评估）
        expected_answer = self.answer_evaluator._get_expected_answer_for_task(last_task_id, patient_profile)
        local_invalid_kind = self.policy._invalid_answer_kind(user_answer)
        local_invalid_skip = False
        if local_invalid_kind:
            invalid_attempt = self.policy._register_invalid_attempt(last_task_id, local_invalid_kind)
            invalid_limit = self.policy._max_invalid_attempts()
            self._log_summary_card(
                "Invalid Answer",
                {
                    "task": last_task_id,
                    "kind": local_invalid_kind,
                    "attempt": f"{invalid_attempt}/{invalid_limit}",
                },
            )
            if invalid_attempt < invalid_limit:
                if local_invalid_kind == "refusal":
                    return self.policy._enter_comfort_after_refusal(
                        last_task_id,
                        user_answer,
                        patient_profile,
                        chat_history,
                        dimension_name,
                        dimension_id,
                        start_time,
                    )
                retry_question = self.policy._build_retry_question(local_invalid_kind, doctor_question, invalid_attempt)
                self.state._last_generated_question = retry_question
                return {
                    'output': retry_question,
                    'response': retry_question,
                    'is_retry': True,
                    'invalid_answer_kind': local_invalid_kind,
                    'invalid_attempt': invalid_attempt,
                    'dimension': dimension_name,
                    'dimension_id': dimension_id,
                    'total_time': time.time() - start_time
                }
            local_invalid_skip = True
            task_max_points = last_task_cfg.get('max_points', 0)
            resistance_result = {
                'is_resistant': local_invalid_kind == 'refusal',
                'confidence': 1.0,
                'category': local_invalid_kind,
                'wants_repeat': local_invalid_kind == 'repeat_request',
                'has_substantive_answer': False,
                'turn_intent': local_invalid_kind,
            }
            eval_result = self.policy._build_zero_score_eval(
                last_task_id,
                local_invalid_kind,
                user_answer,
                task_max_points,
            )
    
        # 🔥 v1优化：三级快速路径 - 越简单的回答走越快的路径
        simple_answers = ["好", "好的", "嗯", "对", "是", "是的", "知道", "明白", "行", "可以"]
        is_simple_confirm = user_answer.strip() in simple_answers
        is_short_answer = len(user_answer.strip()) <= 3
    
        # 🔥 v1: 规则化快速评估（完全跳过 LLM，省 1-4 秒）
        rule_eval_result = None if local_invalid_skip else self.answer_evaluator._try_rule_based_evaluation(
            last_task_id, user_answer, expected_answer
        )
    
        if local_invalid_skip:
            self._log_summary_card(
                "Invalid Answer",
                {
                    "task": last_task_id,
                    "action": "skip_and_score_zero",
                    "kind": local_invalid_kind,
                },
            )
        elif rule_eval_result is not None:
            # ⚡ 极速路径：规则可以直接判断（数字回答、关键词匹配等）
            self._log_verbose(f"极速路径: 规则评估 '{user_answer[:15]}' → {rule_eval_result.get('quality_level')}")
            resistance_result = {
                'is_resistant': False,
                'confidence': 1.0,
                'category': 'none',
                'wants_repeat': False,
                'has_substantive_answer': bool(user_answer.strip()),
                'turn_intent': 'answer',
            }
            self.tool_gateway.cancel_retrieval_prefetch()
            eval_result = rule_eval_result
        elif is_simple_confirm or is_short_answer:
            # 简单确认回答：只做回答评估，跳过抵抗检测
            self._log_verbose(f"快速路径: 简单回答'{user_answer[:10]}'，跳过抵抗检测")
            resistance_result = {
                'is_resistant': False,
                'confidence': 1.0,
                'category': 'none',
                'wants_repeat': False,
                'has_substantive_answer': bool(user_answer.strip()),
                'turn_intent': 'answer',
            }
            self.tool_gateway.cancel_retrieval_prefetch()
            eval_result = self.tool_gateway._call_answer_evaluation(
                doctor_question, user_answer, last_task_id, patient_profile, expected_answer
            )
        else:
            return self._evaluate_cognitive_parallel(
                last_task_id=last_task_id,
                last_task_cfg=last_task_cfg,
                user_answer=user_answer,
                doctor_question=doctor_question,
                patient_profile=patient_profile,
                chat_history=chat_history,
                expected_answer=expected_answer,
                dimension_id=dimension_id,
                dimension_name=dimension_name,
                start_time=start_time,
                step1_start=step1_start,
            )

        return AnswerPathResult(
            expected_answer=expected_answer,
            evaluation=eval_result,
            resistance=resistance_result,
        )

    def _evaluate_cognitive_parallel(
        self,
        *,
        last_task_id: str,
        last_task_cfg: Dict[str, Any],
        user_answer: str,
        doctor_question: str,
        patient_profile: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        expected_answer: Optional[str],
        dimension_id: str,
        dimension_name: str,
        start_time: float,
        step1_start: float,
    ) -> AnswerPathResult | Dict[str, Any]:
        """Run resistance, scoring and next-task prefetch concurrently."""
        invalid_skip_from_resistance = False
        # 🔥 优化：ResistanceTool (API) 与 AnswerEval (Local) 与 TaskSelection (API) 并行执行
        self._log_verbose("阶段一: 并行执行抵抗检测(API)+回答评估(Local)+任务选择(API)")
    
        # 🔥 预计算：本轮结束后任务是否会完成？（用于决定是否需要启动任务选择）
        # 轮数+1（用户回答了这一轮）
        predicted_turns = self.state._task_turns.get(last_task_id, 0) + 1
        min_turns = self.catalog.task_config.get(last_task_id, {}).get('min_turns', 1)
        task_will_complete = (predicted_turns >= min_turns)
    
        # 🔥 预更新 _task_done（供任务选择使用，之后会正式更新）
        # 仅追踪本方法自己推测性添加的标记，便于异常时精确回滚。
        pre_marked_done = False
        if (
            task_will_complete
            and last_task_id
            and last_task_id not in self.state._task_done
        ):
            self.state._task_done.add(last_task_id)
            pre_marked_done = True
            self._log_verbose(f"预测任务完成: {last_task_id} (轮数: {predicted_turns}/{min_turns})")

        # 知识检索是纯函数，提交到常驻线程池跨阶段存活，Phase 3 出题时才消费；
        # 本地线程池只承载必须在本阶段 join 的抵抗/评估/选题。
        self.tool_gateway.start_retrieval_prefetch(dimension_name, chat_history)

        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                # 提交任务
                future_resist = executor.submit(
                    self.tool_gateway._call_resistance_detection,
                    doctor_question,
                    user_answer,
                )
                future_eval = executor.submit(
                    self.tool_gateway._call_answer_evaluation,
                    doctor_question,
                    user_answer,
                    last_task_id, patient_profile, expected_answer
                )

                # 🔥 如果任务会完成，并行启动任务选择
                future_task_select = None
                if task_will_complete and not self.state._pending_consent_task_id:
                    future_task_select = executor.submit(
                        self.task_planner._select_next_task
                    )

                # 获取抵抗检测结果
                resistance_result = future_resist.result()

                resistance_handling = self._handle_parallel_resistance(
                    resistance_result=resistance_result,
                    last_task_id=last_task_id,
                    last_task_cfg=last_task_cfg,
                    user_answer=user_answer,
                    doctor_question=doctor_question,
                    patient_profile=patient_profile,
                    chat_history=chat_history,
                    dimension_id=dimension_id,
                    dimension_name=dimension_name,
                    start_time=start_time,
                    step1_start=step1_start,
                    task_will_complete=task_will_complete,
                    future_task_select=future_task_select,
                )
                if isinstance(resistance_handling, dict):
                    return resistance_handling
                invalid_skip_from_resistance = resistance_handling.skip_evaluation
                if resistance_handling.evaluation is not None:
                    eval_result = resistance_handling.evaluation

                if not invalid_skip_from_resistance and not resistance_result.get('is_resistant'):
                    # 无抵抗或轻微抵抗，获取评估结果
                    eval_result = future_eval.result()

                    # 🔥 如果启动了任务选择，获取结果
                    if future_task_select:
                        self.state._precomputed_next_task = future_task_select.result()

                        # 获取话题提示
                        topic_info = self.state._last_bridge_hint or "N/A"
                        self._log_summary_card(
                            "Task Pre-Selection",
                            {
                                "Task": self.state._precomputed_next_task,
                                "Bridge": topic_info,
                                "Intent": resistance_result.get('turn_intent') or '-',
                            },
                        )
        except Exception:
            # 异常路径：本轮未走到正式评分，撤销推测性的完成标记，
            # 否则该认知任务会被永久跳过而漏分。
            if pre_marked_done:
                self.state._task_done.discard(last_task_id)
                self._log_verbose(f"异常回滚预标记完成: {last_task_id}")
            raise

        return AnswerPathResult(
            expected_answer=expected_answer,
            evaluation=eval_result,
            resistance=resistance_result,
        )

    def _handle_parallel_resistance(
        self,
        *,
        resistance_result: Dict[str, Any],
        last_task_id: str,
        last_task_cfg: Dict[str, Any],
        user_answer: str,
        doctor_question: str,
        patient_profile: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        dimension_id: str,
        dimension_name: str,
        start_time: float,
        step1_start: float,
        task_will_complete: bool,
        future_task_select: Any,
    ) -> ParallelResistanceResult | Dict[str, Any]:
        """Turn repeat/refusal detection into retry, comfort or score-zero outcomes."""
        invalid_skip_from_resistance = False
        # 🔥 处理请求重复：只有明确要求重复且未给出实质回答时，才直接重复上一个问题
        if resistance_result.get('wants_repeat') and not resistance_result.get('has_substantive_answer'):
            invalid_attempt = self.policy._register_invalid_attempt(last_task_id, "repeat_request")
            invalid_limit = self.policy._max_invalid_attempts()
            self._log_summary_card(
                "Repeat Request",
                {
                    "action": "retry_or_skip",
                    "dimension": dimension_name,
                    "task": self.state._last_task_id,
                    "attempt": f"{invalid_attempt}/{invalid_limit}",
                },
            )
            if invalid_attempt < invalid_limit:
                if task_will_complete and last_task_id:
                    self.state._task_done.discard(last_task_id)
                repeated_question = self.policy._build_retry_question("repeat_request", doctor_question, invalid_attempt)
                self.state._last_generated_question = repeated_question
                return {
                    'output': repeated_question,
                    'response': repeated_question,
                    'is_repeat': True,
                    'is_retry': True,
                    'invalid_answer_kind': 'repeat_request',
                    'invalid_attempt': invalid_attempt,
                    'dimension': dimension_name,
                    'dimension_id': dimension_id,
                    'total_time': time.time() - start_time
                }
            eval_result = self.policy._build_zero_score_eval(
                last_task_id,
                "repeat_request",
                user_answer,
                last_task_cfg.get('max_points', 0),
            )
            self._log_summary_card(
                "Invalid Answer",
                {"task": last_task_id, "action": "skip_and_score_zero", "kind": "repeat_request"},
            )
            # 检索预取仍挂在 gateway 上，Phase 3 出题时消费；此处只需收回选题结果。
            if future_task_select:
                self.state._precomputed_next_task = future_task_select.result()
            invalid_skip_from_resistance = True
        if (
            not invalid_skip_from_resistance
            and resistance_result.get('category') == 'repeat_request'
            and resistance_result.get('has_substantive_answer')
        ):
            self._log_verbose("检测到重复倾向，但用户已给出实质回答，不重复上一问题")
        # 🔥 检测到抵抗 -> 直接进入闲聊模式（工具内部已做LLM仲裁，结果可信）
        if not invalid_skip_from_resistance and resistance_result.get('is_resistant'):
            invalid_kind = self.policy._invalid_answer_kind(user_answer, resistance_result)
            if invalid_kind in {"refusal", "unable"}:
                invalid_attempt = self.policy._register_invalid_attempt(last_task_id, invalid_kind)
                invalid_limit = self.policy._max_invalid_attempts()
                self._log_summary_card(
                    "Invalid Answer",
                    {
                        "task": last_task_id,
                        "kind": invalid_kind,
                        "attempt": f"{invalid_attempt}/{invalid_limit}",
                    },
                )
                if invalid_attempt < invalid_limit:
                    if task_will_complete and last_task_id:
                        self.state._task_done.discard(last_task_id)
                    if invalid_kind == "refusal":
                        return self.policy._enter_comfort_after_refusal(
                            last_task_id,
                            user_answer,
                            patient_profile,
                            chat_history,
                            dimension_name,
                            dimension_id,
                            start_time,
                        )
                    retry_question = self.policy._build_retry_question(invalid_kind, doctor_question, invalid_attempt)
                    self.state._last_generated_question = retry_question
                    return {
                        'output': retry_question,
                        'response': retry_question,
                        'is_retry': True,
                        'invalid_answer_kind': invalid_kind,
                        'invalid_attempt': invalid_attempt,
                        'dimension': dimension_name,
                        'dimension_id': dimension_id,
                        'total_time': time.time() - start_time
                    }
                eval_result = self.policy._build_zero_score_eval(
                    last_task_id,
                    invalid_kind,
                    user_answer,
                    last_task_cfg.get('max_points', 0),
                )
                self._log_summary_card(
                    "Invalid Answer",
                    {"task": last_task_id, "action": "skip_and_score_zero", "kind": invalid_kind},
                )
                # 零分路径仍会走到 Phase 3 出题，检索由 gateway 预取在那里消费。
                if future_task_select:
                    self.state._precomputed_next_task = future_task_select.result()
                invalid_skip_from_resistance = True
            if not invalid_skip_from_resistance:
                category = resistance_result.get('category', 'unknown')
                self._log_summary_card(
                    "Resistance Interrupt",
                    {"path": "main", "category": category, "action": "enter_comfort_mode"},
                )
                if task_will_complete and last_task_id:
                    self.state._task_done.discard(last_task_id)

                # 进入闲聊模式
                self.state.is_in_comfort_mode = True
                self.state.comfort_turn_count = 0
                self.state._comfort_entry_category = category
                interrupted_task = self.policy._get_recent_cognitive_task_id(last_task_id)
                if interrupted_task and interrupted_task not in self.catalog.buffer_tasks:
                    self.state._comfort_interrupted_task_id = interrupted_task
                # 调用 comfort 工具生成安抚话术
                comfort_result = self.tool_gateway.comfort_tool._run(
                    resistance_category=category,
                    patient_answer=user_answer,
                    patient_name=patient_profile.get('name'),
                    patient_age=patient_profile.get('age'),
                    patient_gender=patient_profile.get('gender'),
                    used_topics=self.state._used_chat_topics,
                    chat_history=chat_history,  # 🔥 新增：传入聊天记录
                    use_template=False,
                )

                # 解析结果并记录话题
                comfort_data = json.loads(comfort_result)
                if topic := comfort_data.get('selected_topic'):
                    self.state._used_chat_topics.append(topic)
                    self._log_verbose(f"记录新话题: {topic}")

                comfort_message = comfort_data.get('comfort_message', '您说得对，咱们继续聊聊。')
                self.state._last_generated_question = comfort_message
                self._log_summary_card(
                    "Comfort Mode",
                    {
                        "Category": category,
                        "Topic": topic if 'topic' in locals() else '-',
                        "Time": f"{time.time() - step1_start:.2f}s",
                    },
                )

                return {
                    'output': comfort_message,
                    'response': comfort_message,
                    'has_resistance': True,
                    'resistance_category': category,
                    'is_comfort_mode': True,
                    'dimension': dimension_name,
                    'dimension_id': dimension_id,
                    'total_time': time.time() - start_time
                }
        return ParallelResistanceResult(
            skip_evaluation=invalid_skip_from_resistance,
            evaluation=(
                eval_result if invalid_skip_from_resistance else None
            ),
        )
