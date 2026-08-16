from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar


TASK_CONFIG: dict[str, dict[str, Any]] = {
    "persona_collect_1": {
        "dimension_id": None,
        "max_points": 0,
        "type": "buffer",
        "min_turns": 1,
    },
    "persona_collect_2": {
        "dimension_id": None,
        "max_points": 0,
        "type": "buffer",
        "min_turns": 1,
    },
    "buffer_chat": {
        "dimension_id": None,
        "max_points": 0,
        "type": "buffer",
        "min_turns": 1,
    },
    "buffer_consent": {
        "dimension_id": None,
        "max_points": 0,
        "type": "buffer",
        "min_turns": 1,
    },
    "orientation_time_year": {
        "dimension_id": "orientation",
        "max_points": 1,
        "type": "task",
        "min_turns": 1,
    },
    "orientation_time_season": {
        "dimension_id": "orientation",
        "max_points": 1,
        "type": "task",
        "min_turns": 1,
    },
    "orientation_time_month_date": {
        "dimension_id": "orientation",
        "max_points": 2,
        "type": "task",
        "min_turns": 1,
    },
    "orientation_time_weekday": {
        "dimension_id": "orientation",
        "max_points": 1,
        "type": "task",
        "min_turns": 1,
    },
    "orientation_place_province_city": {
        "dimension_id": "orientation",
        "max_points": 2,
        "type": "task",
        "min_turns": 1,
    },
    "orientation_place_district": {
        "dimension_id": "orientation",
        "max_points": 1,
        "type": "task",
        "min_turns": 1,
    },
    "orientation_place_location_floor": {
        "dimension_id": "orientation",
        "max_points": 2,
        "type": "task",
        "min_turns": 1,
    },
    "registration_3words": {
        "dimension_id": "registration",
        "max_points": 3,
        "type": "task",
        "min_turns": 1,
    },
    "attention_calc_life_math": {
        "dimension_id": "attention_calculation",
        "max_points": 5,
        "type": "task",
        "min_turns": 1,
    },
    "attention_reverse_phrase": {
        "dimension_id": "attention_calculation",
        "max_points": 5,
        "type": "task",
        "min_turns": 1,
    },
    "language_naming_watch": {
        "dimension_id": "language",
        "max_points": 1,
        "type": "task",
        "min_turns": 1,
    },
    "language_naming_pencil": {
        "dimension_id": "language",
        "max_points": 1,
        "type": "task",
        "min_turns": 1,
    },
    "language_repetition_sentence": {
        "dimension_id": "language",
        "max_points": 1,
        "type": "task",
        "min_turns": 1,
    },
    "language_3step_action": {
        "dimension_id": "language",
        "max_points": 3,
        "type": "task",
        "min_turns": 1,
    },
    "language_reading_close_eyes": {
        "dimension_id": "language",
        "max_points": 1,
        "type": "task",
        "min_turns": 1,
    },
    "language_writing_sentence": {
        "dimension_id": "language",
        "max_points": 1,
        "type": "task",
        "min_turns": 1,
    },
    "recall_3words": {
        "dimension_id": "recall",
        "max_points": 3,
        "type": "task",
        "min_turns": 1,
    },
    "copy_pentagons": {
        "dimension_id": "copy",
        "max_points": 1,
        "type": "task",
        "min_turns": 1,
    },
}

REQUIRED_TASKS: tuple[str, ...] = (
    "persona_collect_1",
    "persona_collect_2",
    "orientation_time_year",
    "orientation_time_season",
    "orientation_time_month_date",
    "orientation_time_weekday",
    "orientation_place_province_city",
    "orientation_place_district",
    "orientation_place_location_floor",
    "registration_3words",
    "attention_calc_life_math",
    "attention_reverse_phrase",
    "language_naming_watch",
    "language_naming_pencil",
    "language_repetition_sentence",
    "language_3step_action",
    "language_reading_close_eyes",
    "language_writing_sentence",
    "recall_3words",
    "copy_pentagons",
)

BUFFER_TASKS = frozenset(
    {
        "persona_collect_1",
        "persona_collect_2",
        "buffer_chat",
        "buffer_consent",
        "buffer_answer_question",
    }
)

VISUAL_ACTION_TASKS = frozenset()

TASK_KEYWORD_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("orientation_time_year", ("哪一年", "年份", "今年")),
    ("orientation_time_season", ("季节", "什么季")),
    ("orientation_time_month_date", ("几月", "几号", "日期", "月份", "哪天")),
    ("orientation_time_weekday", ("星期", "周几", "礼拜")),
    ("orientation_place_province_city", ("省", "城市", "哪个市", "什么市")),
    ("orientation_place_district", ("哪个区", "什么区", "什么县", "哪个县")),
    (
        "orientation_place_location_floor",
        ("什么地方", "哪里", "几楼", "几层", "这是哪"),
    ),
    ("persona_collect_1", ("爱好", "兴趣", "喜欢做", "喜欢什么", "娱乐", "消遣")),
    ("persona_collect_2", ("起床", "睡觉", "作息", "吃什么", "早饭", "晚饭", "习惯")),
    ("registration_3words", ("记住", "记下", "这几个词", "三个词")),
    ("recall_3words", ("刚才", "记的词", "还记得", "之前的词")),
    ("language_naming_watch", ("手表", "看图")),
    ("language_naming_pencil", ("铅笔",)),
    ("language_repetition_sentence", ("复述", "跟我说", "再说一遍")),
    ("language_reading_close_eyes", ("闭眼", "读字", "闭上眼")),
    (
        "language_3step_action",
        ("三步", "右手", "左手", "举手", "握拳", "拳头", "胸前", "指令"),
    ),
    ("language_writing_sentence", ("说句子", "说一个完整", "完整句子", "说一句")),
    ("attention_calc_life_math", ("算", "减7", "减去", "计算", "100")),
    (
        "attention_reverse_phrase",
        ("倒着说", "倒着念", "倒过来", "倒转", "祝", "出", "入", "平", "安"),
    ),
    ("copy_pentagons", ("画", "临摹", "五边形", "照着画")),
)

TOPIC_CATEGORIES: dict[str, tuple[str, ...]] = {
    "饮食": (
        "吃",
        "饭",
        "菜",
        "早餐",
        "午餐",
        "晚餐",
        "早饭",
        "午饭",
        "晚饭",
        "喝",
        "食",
        "饮",
        "零食",
        "水果",
        "做饭",
        "厨房",
        "菜市场",
    ),
    "住所": ("住", "城市", "区", "地方", "家", "房", "搬", "小区", "街", "楼"),
    "天气": ("天气", "冷", "热", "下雨", "下雪", "风", "温度", "晴", "阴"),
    "兴趣爱好": (
        "爱好",
        "喜欢",
        "兴趣",
        "看书",
        "钓鱼",
        "打牌",
        "下棋",
        "跳舞",
        "唱歌",
        "养花",
        "散步",
        "运动",
        "锻炼",
        "太极",
    ),
    "日常活动": ("今天做", "干什么", "安排", "活动", "出门", "去哪", "逛"),
    "家人": ("家人", "孩子", "儿子", "女儿", "老伴", "孙", "媳妇"),
    "健康": ("身体", "睡", "觉", "锻炼", "医院", "药", "腿", "腰", "眼睛"),
    "电视节目": ("电视", "节目", "新闻", "看", "剧", "频道"),
    "时间": ("星期", "周", "几号", "月", "季节", "日期", "今天"),
}

DIMENSION_QUERY_MAP: dict[str, str] = {
    "定向力": "阿尔茨海默病 定向力 时间地点 认知评估",
    "即时记忆": "阿尔茨海默病 即时记忆 三词登记 认知评估",
    "注意力与计算": "阿尔茨海默病 注意力 计算 连续减法 认知评估",
    "延迟回忆": "阿尔茨海默病 延迟回忆 记忆 认知评估",
    "语言": "阿尔茨海默病 语言 命名复述 认知评估",
    "构图(临摹)": "阿尔茨海默病 构图 临摹 视觉空间 认知评估",
}


@dataclass(frozen=True)
class ScreeningTaskCatalog:
    """Immutable task metadata and small domain queries for screening flows."""

    task_config: ClassVar[dict[str, dict[str, Any]]] = TASK_CONFIG
    required_tasks: ClassVar[tuple[str, ...]] = REQUIRED_TASKS
    buffer_tasks: ClassVar[frozenset[str]] = BUFFER_TASKS
    visual_action_tasks: ClassVar[frozenset[str]] = VISUAL_ACTION_TASKS
    task_keyword_rules: ClassVar[
        tuple[tuple[str, tuple[str, ...]], ...]
    ] = TASK_KEYWORD_RULES
    topic_categories: ClassVar[dict[str, tuple[str, ...]]] = TOPIC_CATEGORIES
    dimension_query_map: ClassVar[dict[str, str]] = DIMENSION_QUERY_MAP

    def config_for(self, task_id: str | None) -> dict[str, Any]:
        return self.task_config.get(task_id or "", {})

    def is_buffer(self, task_id: str | None) -> bool:
        return bool(task_id and task_id in self.buffer_tasks)

    def remaining_tasks(self, completed: set[str]) -> list[str]:
        return [task for task in self.required_tasks if task not in completed]
