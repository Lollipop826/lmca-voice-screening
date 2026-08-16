from __future__ import annotations

from collections.abc import Callable
import json
import time
from typing import Any, Dict, List

from .answer_evaluation import ScreeningAnswerEvaluation
from .catalog import ScreeningTaskCatalog
from .conversation_policy import ScreeningConversationPolicy
from .question_generation import ScreeningQuestionGeneration
from .state import ScreeningSessionState
from .task_planning import ScreeningTaskPlanning
from .tool_gateway import ScreeningToolGateway


class ScreeningComfortPhase:
    """Handle a turn while the screening conversation is in comfort mode."""

    def __init__(
        self,
        *,
        state: ScreeningSessionState,
        catalog: ScreeningTaskCatalog,
        task_planner: ScreeningTaskPlanning,
        tools: ScreeningToolGateway,
        policy: ScreeningConversationPolicy,
        answer_evaluator: ScreeningAnswerEvaluation,
        question_generator: ScreeningQuestionGeneration,
        dimension_map: Dict[str, Dict[str, Any]],
        log_summary: Callable[[str, Dict[str, Any]], None],
        log_verbose: Callable[[str], None],
    ) -> None:
        self.state = state
        self.catalog = catalog
        self.task_planner = task_planner
        self.tool_gateway = tools
        self.policy = policy
        self.answer_evaluator = answer_evaluator
        self.question_generator = question_generator
        self.dimension_map = dimension_map
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

    def handle(
        self,
        *,
        user_answer: str,
        session_id: str,
        patient_profile: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        current_emotion: str,
        dimension_id: str,
        dimension_name: str,
        start_time: float,
    ) -> Dict[str, Any]:
        """Handle a complete comfort-mode turn and return immediately."""
        # ==================== 闲聊模式：不检测抵抗，只管理闲聊轮次 ====================
        self.state.comfort_turn_count += 1
        
        # 🔔 检测用户是否要结束对话（goodbye检测）
        goodbye_keywords = ['结束', '再见', '拜拜', '不聊了', '就这样', '到这儿', '先这样', '回头聊']
        is_goodbye = any(kw in user_answer for kw in goodbye_keywords)
        
        if is_goodbye:  #关键词监测到执行
            # 🔍 检查 MMSE 是否已完成所有维度
            mmse_complete = self.policy._check_mmse_complete(session_id)
            
            if mmse_complete:
                # ✅ MMSE 已完成，可以正常告别
                self._log_summary_card(
                    "Comfort Exit",
                    {"trigger": "goodbye", "mmse": "complete", "action": "farewell"},
                )
                farewell = f"{patient_profile.get('name', '')}，好嘞，那今天就到这儿，您好好休息。想聊的时候随时叫我！"
                self.state._last_generated_question = farewell

                
                return {
                    'output': farewell,
                    'response': farewell,
                    'is_comfort_mode': True,
                    'is_goodbye': True,
                    'total_time': time.time() - start_time
                }
            else:
                # ⚠️ MMSE 未完成，不能真正结束，用 LLM 生成自然的过渡
                self._log_summary_card(
                    "Comfort Exit",
                    {"trigger": "goodbye", "mmse": "incomplete", "action": "soft_continue"},
                )
                soft_continue = self.policy._generate_soft_continuation(
                    patient_profile, user_answer, chat_history
                )
                self.state._last_generated_question = soft_continue

                
                return {
                    'output': soft_continue,
                    'response': soft_continue,
                    'is_comfort_mode': True,
                    'is_goodbye': False,
                    'total_time': time.time() - start_time
                }
        
        ask_to_continue = False
        resume_reason = "-"
        positive_emotions = ['happy', 'excited', 'joy', 'positive', 'neutral']
        is_positive = current_emotion in positive_emotions
        has_substantive_answer = self.policy._is_substantive_comfort_reply(user_answer)
        min_resume_turns = self.policy._get_comfort_resume_min_turns()
        
        if is_positive and has_substantive_answer and self.state.comfort_turn_count >= min_resume_turns:
            ask_to_continue = True
            resume_reason = "substantive_recovered"
        elif is_positive and self.state.comfort_turn_count >= 3:
            ask_to_continue = True
            resume_reason = "comfort_cap"

        self._log_summary_card(
            "Comfort Mode",
            {
                "turn": self.state.comfort_turn_count,
                "entry_category": self.state._comfort_entry_category or 'normal',
                "min_resume_turns": min_resume_turns,
                "emotion": current_emotion,
                "substantive": has_substantive_answer,
                "decision": "resume_eval" if ask_to_continue else "continue_chat",
                "reason": resume_reason if ask_to_continue else '-',
            },
        )
        
        # 🔥 修复：满足退出条件时，恢复被打断的任务
        if ask_to_continue:
            self.state.is_in_comfort_mode = False
            self.state.comfort_turn_count = 0
            self.state._comfort_entry_category = None
            
            # 🔥 检查被打断的任务是否已完成
            interrupted_task = self.state._comfort_interrupted_task_id
            if interrupted_task and interrupted_task not in self.state._task_done:
                # 继续被打断的任务
                self._log_summary_card(
                    "Comfort Resume",
                    {"action": "resume_interrupted_task", "task": interrupted_task},
                )
                self.state._comfort_interrupted_task_id = None  # 清除记录
                
                # 获取任务配置
                task_cfg = self.catalog.task_config.get(interrupted_task, {})
                task_dim_id = task_cfg.get('dimension_id', dimension_id)
                task_dim_name = self.dimension_map.get(task_dim_id, {}).get('name', dimension_name)
                
                return self.question_generator._generate_assessment_question(
                    task_dim_name, patient_profile, chat_history, start_time,
                    task_id=interrupted_task  # 🔥 传入任务ID
                )
            else:
                # 被打断的任务已完成或不存在，选新任务
                self._log_summary_card("Comfort Resume", {"action": "select_new_task"})
                self.state._comfort_interrupted_task_id = None
                next_task_id = self.task_planner._select_next_task()
                
                if next_task_id is None:
                    # 所有任务完成，生成完成消息
                    summary_json = self.tool_gateway.mmse_tool._run(
                        session_id=session_id,
                        dimension_id="",
                        action="summary",
                    )
                    summary = json.loads(summary_json)
                    scaled_score = summary.get('scaled_total_score', summary.get('total_score', 0))
                    risk_assessment = self.answer_evaluator._calculate_alzheimers_risk(int(scaled_score))
                    completion_message = self.policy._generate_completion_message(int(scaled_score), risk_assessment, patient_profile)


                    return {
                        'output': completion_message,
                        'response': completion_message,
                        'assessment_complete': True,
                        'total_score': summary.get('total_score', 0),
                        'scaled_score': scaled_score,
                        'total_time': time.time() - start_time
                    }
                
                # 获取新任务的维度
                task_cfg = self.catalog.task_config.get(next_task_id, {})
                task_dim_id = task_cfg.get('dimension_id', dimension_id)
                task_dim_name = self.dimension_map.get(task_dim_id, {}).get('name', dimension_name)
                
                return self.question_generator._generate_assessment_question(
                    task_dim_name, patient_profile, chat_history, start_time,
                    task_id=next_task_id  # 🔥 传入任务ID
                )
        
        # 使用原有comfort工具生成闲聊回复
        # 映射情绪到抵抗类别
        emotion_to_category = {
            'sad': 'fatigue',
            'angry': 'hostility',
            'fear': 'avoidance',
            'neutral': 'normal',
            'happy': 'normal',
            'excited': 'normal',
        }
        category = emotion_to_category.get(current_emotion, 'normal')
        comfort_result = self.tool_gateway.comfort_tool._run(
            resistance_category=category,
            patient_answer=user_answer,
            patient_name=patient_profile.get('name'),
            patient_age=patient_profile.get('age'),
            patient_gender=patient_profile.get('gender'),  # 🔥 新增：性别参数
            used_topics=self.state._used_chat_topics,  # 🔥 传入已聊话题
            chat_history=chat_history,  # 🔥 新增：传入聊天记录
            use_template=False,
        )
        
        # 更新已聊话题
        try:
            comfort_data = json.loads(comfort_result)
            if topic := comfort_data.get('selected_topic'):
                self.state._used_chat_topics.append(topic)
                self._log_verbose(f"记录新话题: {topic} (总计: {len(self.state._used_chat_topics)})")
        except:
            pass
        comfort_data = json.loads(comfort_result)
        comfort_message = comfort_data.get('comfort_message', '您说得对，咱们继续聊聊。')
        self.state._last_generated_question = comfort_message
        self.state._last_task_id = "buffer_chat"
        
        # 🔥 记录闲聊问题到 _asked_questions（防止退出闲聊后重复）
        if comfort_message and len(comfort_message) > 5:
            self.state._asked_questions.append(comfort_message[:80])
            if len(self.state._asked_questions) > 20:
                self.state._asked_questions = self.state._asked_questions[-20:]
        
        return {
            'output': comfort_message,
            'response': comfort_message,
            'is_comfort_mode': True,
            'comfort_turn': self.state.comfort_turn_count,
            'total_time': time.time() - start_time
        }
