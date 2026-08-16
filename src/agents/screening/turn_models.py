from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class EvaluationPhaseResult:
    last_task_id: Optional[str]
    expected_answer: Optional[str]
    evaluation: dict[str, Any]
    elapsed: float


@dataclass(frozen=True)
class AnswerPathResult:
    expected_answer: Optional[str]
    evaluation: dict[str, Any]
    resistance: dict[str, Any]


@dataclass(frozen=True)
class ParallelResistanceResult:
    skip_evaluation: bool
    evaluation: Optional[dict[str, Any]]


@dataclass(frozen=True)
class TaskChoice:
    next_task_id: Optional[str]
    evaluation: dict[str, Any]
    mmse_score: dict[str, Any]
    dimension_id: str
    dimension_name: str


@dataclass(frozen=True)
class RoutingPhaseResult:
    next_task_id: str
    dimension_id: str
    dimension_name: str
    evaluation: dict[str, Any]
    mmse_score: dict[str, Any]
    elapsed: float


@dataclass(frozen=True)
class QuestionPhaseResult:
    question: str
    effective_task_id: str
    image_display: Optional[dict[str, Any]]
    vision_command: Optional[dict[str, Any]]
    elapsed: float


@dataclass(frozen=True)
class QuestionDraft:
    question: Optional[str]
    effective_task_id: str
    image_display: Optional[dict[str, Any]] = None
    vision_command: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class TurnContext:
    start_time: float
    session_id: str
    patient_profile: dict[str, Any]
    chat_history: list[dict[str, str]]
    current_emotion: str
    dimension_id: str
    dimension_name: str
    doctor_question: str
    user_answer: str


TASK_DESCRIPTIONS = {
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

SPECIAL_QUESTION_TASKS = {
    "registration_3words",
    "recall_3words",
    "attention_calc_life_math",
    "attention_reverse_phrase",
    "language_naming_watch",
    "language_naming_pencil",
    "language_repetition_sentence",
    "language_reading_close_eyes",
    "language_3step_action",
}

BRIDGED_SPECIAL_TASKS = {
    "registration_3words",
    "recall_3words",
    "attention_calc_life_math",
    "attention_reverse_phrase",
    "language_naming_watch",
    "language_naming_pencil",
    "language_repetition_sentence",
}

TASK_TO_SPECIAL_DIMENSION = {
    "registration_3words": "registration",
    "recall_3words": "recall",
    "attention_calc_life_math": "attention_calculation",
    "attention_reverse_phrase": "attention_reverse_phrase",
    "language_repetition_sentence": "language_repetition",
}
