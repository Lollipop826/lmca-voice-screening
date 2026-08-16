"""
MMSE评分工具 - 适配本地35分量表

当前计分方式：
1. 定向力（Orientation）：10分（时间5分 + 地点5分）
2. 即时记忆（Registration）：3分
3. 注意力与计算（Attention & Calculation）：10分（连续减法5分 + 倒着说词组5分）
4. 延迟回忆（Recall）：3分
5. 语言（Language）：8分
6. 构图（Copy）：1分

原始总分：35分
折算总分：保留30分口径，便于沿用传统MMSE解释阈值
"""

from __future__ import annotations

import os
import json
import re
from typing import Optional, Type, Dict, Any, List
from pathlib import Path
from datetime import datetime

from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool
from src.db.database import (
    is_cognitive_screening_session as db_is_cognitive_screening_session,
    save_mmse_score as db_save_mmse_score,
)


# MMSE标准分值
MMSE_STANDARD_SCORES = {
    "orientation": 10,          # 定向力：时间5分 + 地点5分
    "registration": 3,          # 即时记忆：3个词
    "attention_calculation": 10, # 注意力与计算：100-7连续5次 + 倒着说词组
    "recall": 3,                # 延迟回忆：3个词
    "language": 8,              # 语言：命名2+复述1+三步指令3+阅读1+书写1
    "copy": 1,                  # 构图：临摹五边形
}

MMSE_TOTAL_SCORE = 35
MMSE_REFERENCE_SCORE = 30

MMSE_TASK_LABELS = {
    "orientation_time_year": "年份",
    "orientation_time_season": "季节",
    "orientation_time_month_date": "月份/日期",
    "orientation_time_weekday": "星期",
    "orientation_place_province_city": "省/市",
    "orientation_place_district": "区/县",
    "orientation_place_location_floor": "医院/楼层",
    "registration_3words": "三词登记",
    "attention_calc_life_math": "连续减法",
    "attention_reverse_phrase": "倒着说词组",
    "recall_3words": "三词回忆",
    "language_naming_watch": "命名(手表)",
    "language_naming_pencil": "命名(铅笔)",
    "language_repetition_sentence": "复述",
    "language_3step_action": "3级命令",
    "language_reading_close_eyes": "阅读",
    "language_writing_sentence": "说句子",
    "copy_pentagons": "图形临摹",
}

MMSE_TASK_ORDER = {
    "orientation_time_year": 10,
    "orientation_time_season": 20,
    "orientation_time_month_date": 30,
    "orientation_time_weekday": 40,
    "orientation_place_province_city": 50,
    "orientation_place_district": 60,
    "orientation_place_location_floor": 70,
    "registration_3words": 80,
    "attention_calc_life_math": 90,
    "attention_reverse_phrase": 95,
    "recall_3words": 100,
    "language_naming_watch": 110,
    "language_naming_pencil": 120,
    "language_repetition_sentence": 130,
    "language_3step_action": 140,
    "language_reading_close_eyes": 150,
    "language_writing_sentence": 160,
    "copy_pentagons": 170,
}


class MMSEScoringToolArgs(BaseModel):
    """MMSE评分工具参数"""
    
    session_id: str = Field(..., description="会话ID")
    dimension_id: str = Field(..., description="维度ID（orientation/registration/attention_calculation/recall/language/copy，支持灵活任务如orientation_assessment/language_assessment）")
    score: int = Field(..., description="该维度得分（0到该维度满分）")
    max_score: Optional[int] = Field(None, description="该维度满分（可选，默认使用标准分值）")
    task_id: Optional[str] = Field(None, description="任务ID；同一 task_id 会锁定为该任务的最佳得分")
    question: str = Field(default="", description="评估问题")
    answer: str = Field(default="", description="患者回答")
    evaluation_detail: str = Field(default="", description="评分依据")
    action: str = Field(default="save", description="操作类型：save（保存）/get（获取）/summary（汇总）")
    education_years: Optional[int] = Field(None, description="患者受教育年限，用于按教育水平解释MMSE总分")
    education_level: Optional[str] = Field(None, description="患者教育水平：文盲/小学/初中")


class MMSEScoringResult(BaseModel):
    """MMSE评分结果"""
    
    success: bool = Field(..., description="操作是否成功")
    message: str = Field(..., description="操作结果消息")
    dimension_score: Optional[int] = Field(None, description="当前维度得分")
    dimension_max_score: Optional[int] = Field(None, description="当前维度满分")
    total_score: Optional[int] = Field(None, description="累计总分")
    total_max_score: int = Field(MMSE_TOTAL_SCORE, description="MMSE总分")
    completed_max_score: Optional[int] = Field(None, description="已评估项目的满分合计（用于折算）")
    scaled_total_score: Optional[float] = Field(None, description="按已评估项目折算到30分制的总分")
    coverage: Optional[float] = Field(None, description="覆盖率=已评估满分/35")
    missing_dimensions: Optional[List[str]] = Field(None, description="未评估的维度列表")
    cognitive_status: Optional[str] = Field(None, description="认知功能状态")
    completed_dimensions: Optional[List[str]] = Field(None, description="已完成的维度")
    scoring_details: Optional[Dict[str, Any]] = Field(None, description="详细评分记录")


class MMSEScoringTool(BaseTool):
    """
    MMSE标准评分工具
    
    按照本地35分量表记录评分，并同时输出折算到30分制的解释分数。
    """
    
    name: str = "mmse_scoring_tool"
    description: str = (
        "MMSE评分工具（原始35分制，提供折算30分解释分）。"
        "输入参数：session_id（会话ID），dimension_id（维度ID），score（得分），action（操作类型）。"
        "支持操作：save（保存评分）、get（获取维度评分）、summary（获取总分和认知状态）。"
        "维度ID：orientation/registration/attention_calculation/recall/language/copy（支持灵活任务如orientation_assessment/language_assessment）。"
    )
    
    args_schema: Type[BaseModel] = MMSEScoringToolArgs
    
    _scoring_dir: Path = PrivateAttr()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._scoring_dir = Path("data/mmse_scores")
        self._scoring_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _split_task_id(task_id: Optional[str]) -> tuple[str, str]:
        raw = str(task_id or "").strip()
        if "::" in raw:
            base_task_id, suffix = raw.split("::", 1)
            return base_task_id, suffix
        return raw, ""

    def _format_task_label(self, task_id: Optional[str]) -> str:
        base_task_id, suffix = self._split_task_id(task_id)
        label = MMSE_TASK_LABELS.get(base_task_id, base_task_id or str(task_id or ""))
        if base_task_id == "attention_calc_life_math" and suffix:
            if suffix.startswith("turn_"):
                return f"{label} 第{suffix.replace('turn_', '')}步"
            if suffix.isdigit():
                return f"{label}（结果 {suffix}）"
            return f"{label}（{suffix}）"
        return label

    @staticmethod
    def _clean_evaluation_detail(detail: Optional[str]) -> str:
        text = str(detail or "").strip()
        if not text:
            return ""
        text = re.sub(r'^质量等级:\s*[^|｜]+\s*[|｜]\s*', '', text)
        return text.strip()

    def _task_sort_key(self, task_id: Optional[str]) -> tuple[int, int, str]:
        base_task_id, suffix = self._split_task_id(task_id)
        order = MMSE_TASK_ORDER.get(base_task_id, 999)
        if base_task_id == "attention_calc_life_math":
            if suffix.isdigit():
                return (order, -int(suffix), "")
            if suffix.startswith("turn_"):
                try:
                    return (order, int(suffix.replace("turn_", "")), "")
                except ValueError:
                    return (order, 999, suffix)
        return (order, 0, suffix)

    @staticmethod
    def _score_status(score: int, max_score: int) -> str:
        if max_score <= 0:
            return "pending"
        if score >= max_score:
            return "correct"
        if score > 0:
            return "partial"
        return "incorrect"

    @staticmethod
    def _scale_to_reference_score(total_score: int, completed_max_score: int) -> float:
        if completed_max_score <= 0:
            return 0.0
        return round(total_score / completed_max_score * MMSE_REFERENCE_SCORE, 1)

    @staticmethod
    def _resolve_education_band(
        education_years: Optional[int] = None,
        education_level: Optional[str] = None,
    ) -> Dict[str, Any]:
        """按项目约定将教育水平映射到MMSE正常界值（30分口径）。"""
        level_text = str(education_level or "").strip()
        if level_text:
            if "文盲" in level_text or "未受教育" in level_text:
                return {"key": "illiterate", "label": "文盲", "normal_threshold": 19}
            if "小学" in level_text:
                return {"key": "primary", "label": "小学", "normal_threshold": 21}
            if "初中" in level_text or "高中" in level_text or "中专" in level_text or "大学" in level_text or "本科" in level_text:
                return {"key": "middle", "label": "初中及以上", "normal_threshold": 25}

        try:
            years = int(education_years) if education_years is not None else 6
        except (TypeError, ValueError):
            years = 6
        if years <= 0:
            return {"key": "illiterate", "label": "文盲", "normal_threshold": 19}
        if years <= 6:
            return {"key": "primary", "label": "小学", "normal_threshold": 21}
        return {"key": "middle", "label": "初中及以上", "normal_threshold": 25}

    def _build_cognitive_interpretation(
        self,
        total_score: float,
        education_years: Optional[int] = None,
        education_level: Optional[str] = None,
    ) -> Dict[str, Any]:
        band = self._resolve_education_band(education_years, education_level)
        normal_threshold = band["normal_threshold"]
        if total_score >= normal_threshold:
            status = "认知功能正常"
            severity = "normal"
        elif total_score >= 10:
            status = "轻度认知障碍（MCI）"
            severity = "mci"
        else:
            status = "中重度认知障碍"
            severity = "moderate_severe"
        return {
            "status": status,
            "severity": severity,
            "education_level": band["label"],
            "education_key": band["key"],
            "normal_threshold": normal_threshold,
            "score_basis": "30分折算口径",
        }
    
    def _run(
        self,
        session_id: str,
        dimension_id: str,
        score: int = 0,
        max_score: Optional[int] = None,
        task_id: Optional[str] = None,
        question: str = "",
        answer: str = "",
        evaluation_detail: str = "",
        action: str = "save",
        education_years: Optional[int] = None,
        education_level: Optional[str] = None,
    ) -> str:
        """
        执行MMSE评分操作
        
        Returns:
            JSON格式的评分结果
        """
        # 验证 session_id
        if not session_id or not session_id.strip():
            return json.dumps({
                "success": False,
                "message": "session_id 不能为空"
            }, ensure_ascii=False)
        if not db_is_cognitive_screening_session(session_id):
            return json.dumps({
                "success": False,
                "message": "MMSE评分仅允许在认知筛查专项会话中记录",
                "blocked_reason": "NON_COGNITIVE_SESSION",
            }, ensure_ascii=False)
        
        scoring_file = self._scoring_dir / f"{session_id}_mmse.json"
        
        # 读取现有评分
        scoring_data = self._load_scoring(scoring_file)
        
        if action == "save":
            return self._save_score(
                scoring_data, scoring_file, session_id, dimension_id,
                score, max_score, task_id, question, answer, evaluation_detail
            )
        elif action == "get":
            return self._get_score(scoring_data, dimension_id)
        elif action == "summary":
            return self._get_summary(scoring_data, education_years, education_level)
        else:
            return json.dumps({
                "success": False,
                "message": f"不支持的操作类型: {action}"
            }, ensure_ascii=False)
    
    def _load_scoring(self, scoring_file: Path) -> Dict[str, Any]:
        """加载MMSE评分记录"""
        if scoring_file.exists():
            try:
                with open(scoring_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[MMSE] 加载评分文件失败: {type(e).__name__}")
                return self._init_scoring_data()
        else:
            return self._init_scoring_data()
    
    def _init_scoring_data(self) -> Dict[str, Any]:
        """初始化MMSE评分数据结构"""
        return {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "dimensions": {},
            "total_score": 0,
            "total_max_score": MMSE_TOTAL_SCORE,
            "cognitive_status": "未评估"
        }
    
    def _normalize_dimension_entry(self, dimension_id: str, dimension_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """将旧版/新版维度数据统一为 task 聚合结构。"""
        standard_max_score = MMSE_STANDARD_SCORES.get(dimension_id, 0)
        data = dict(dimension_data or {})
        raw_tasks = data.get("tasks")
        tasks: Dict[str, Dict[str, Any]] = {}

        if isinstance(raw_tasks, dict):
            for raw_task_id, raw_task_data in raw_tasks.items():
                if not isinstance(raw_task_data, dict):
                    continue
                task_key = str(raw_task_data.get("task_id") or raw_task_id or dimension_id)
                task_score = int(raw_task_data.get("score", 0) or 0)
                task_max = int(raw_task_data.get("max_score", 0) or 0)
                task_max = min(task_max, standard_max_score) if standard_max_score else task_max
                task_score = min(task_score, task_max) if task_max else 0
                tasks[task_key] = {
                    **raw_task_data,
                    "task_id": task_key,
                    "score": task_score,
                    "max_score": task_max,
                    "records": list(raw_task_data.get("records") or []),
                }
        else:
            legacy_score = int(data.get("score", 0) or 0)
            legacy_max = int(data.get("max_score", 0) or 0)
            legacy_records = list(data.get("records") or [])
            if legacy_score or legacy_max or legacy_records:
                legacy_task_id = str(data.get("task_id") or dimension_id)
                legacy_max = min(legacy_max, standard_max_score) if standard_max_score else legacy_max
                legacy_score = min(legacy_score, legacy_max) if legacy_max else 0
                tasks[legacy_task_id] = {
                    "task_id": legacy_task_id,
                    "score": legacy_score,
                    "max_score": legacy_max,
                    "question": data.get("question", ""),
                    "answer": data.get("answer", ""),
                    "evaluation_detail": data.get("evaluation_detail", ""),
                    "timestamp": data.get("timestamp"),
                    "records": legacy_records,
                }

        dimension_score = min(
            sum(int(task.get("score", 0) or 0) for task in tasks.values()),
            standard_max_score,
        ) if standard_max_score else sum(int(task.get("score", 0) or 0) for task in tasks.values())
        dimension_max_score = min(
            sum(int(task.get("max_score", 0) or 0) for task in tasks.values()),
            standard_max_score,
        ) if standard_max_score else sum(int(task.get("max_score", 0) or 0) for task in tasks.values())
        if dimension_max_score:
            dimension_score = min(dimension_score, dimension_max_score)

        records: List[Dict[str, Any]] = []
        for task_data in tasks.values():
            records.extend(list(task_data.get("records") or []))

        return {
            **data,
            "score": dimension_score,
            "max_score": dimension_max_score,
            "standard_max_score": standard_max_score,
            "tasks": tasks,
            "records": records,
        }

    def _recalculate_totals(self, scoring_data: Dict[str, Any]) -> None:
        total_score = sum(int(d.get("score", 0) or 0) for d in scoring_data["dimensions"].values())
        completed_max_score = sum(
            int(d.get("max_score", d.get("standard_max_score", 0)) or 0)
            for d in scoring_data["dimensions"].values()
        )
        completed_standard_max_score = sum(
            int(d.get("standard_max_score", 0) or 0) for d in scoring_data["dimensions"].values()
        )

        scoring_data["total_score"] = total_score
        scoring_data["completed_max_score"] = completed_max_score
        scoring_data["completed_standard_max_score"] = completed_standard_max_score
        scoring_data["updated_at"] = datetime.now().isoformat()

        scaled_reference_total = self._scale_to_reference_score(total_score, completed_max_score)

        if completed_max_score == MMSE_TOTAL_SCORE:
            scoring_data["cognitive_status"] = self._judge_cognitive_status(scaled_reference_total)
        else:
            scoring_data["cognitive_status"] = f"预估: {self._judge_cognitive_status(scaled_reference_total)} (覆盖率{completed_max_score}/{MMSE_TOTAL_SCORE})"

    def _save_score(
        self,
        scoring_data: Dict[str, Any],
        scoring_file: Path,
        session_id: str,
        dimension_id: str,
        score: int,
        max_score: Optional[int],
        task_id: Optional[str],
        question: str,
        answer: str,
        evaluation_detail: str
    ) -> str:
        """保存MMSE评分"""
        try:
            # 获取该维度的标准分值
            standard_max_score = MMSE_STANDARD_SCORES.get(dimension_id, 0)
            if max_score is None:
                max_score = standard_max_score
            
            # 验证维度ID
            if standard_max_score == 0:
                return json.dumps({
                    "success": False,
                    "message": f"无效的维度ID: {dimension_id}"
                }, ensure_ascii=False)
            
            # 验证分数范围
            if score < 0 or score > max_score:
                return json.dumps({
                    "success": False,
                    "message": f"分数超出范围（0-{max_score}）"
                }, ensure_ascii=False)

            dimension_entry = self._normalize_dimension_entry(
                dimension_id,
                scoring_data["dimensions"].get(dimension_id)
            )
            task_key = (task_id or dimension_id or "").strip() or dimension_id
            existing_task = (dimension_entry.get("tasks") or {}).get(task_key)
            timestamp = datetime.now().isoformat()
            record_item = {
                "task_id": task_key,
                "score": score,
                "max_score": max_score,
                "question": question,
                "answer": answer,
                "evaluation_detail": evaluation_detail,
                "timestamp": timestamp,
            }

            task_records = list(existing_task.get("records") or []) if existing_task else []
            task_records.append(record_item)

            locked_task_score = max(int(existing_task.get("score", 0) or 0), score) if existing_task else score
            locked_task_max = max(int(existing_task.get("max_score", 0) or 0), max_score) if existing_task else max_score
            locked_task_max = min(locked_task_max, standard_max_score)
            locked_task_score = min(locked_task_score, locked_task_max)

            tasks = dict(dimension_entry.get("tasks") or {})
            tasks[task_key] = {
                **(existing_task or {}),
                "task_id": task_key,
                "score": locked_task_score,
                "max_score": locked_task_max,
                "question": question,
                "answer": answer,
                "evaluation_detail": evaluation_detail,
                "timestamp": timestamp,
                "records": task_records,
            }

            all_records: List[Dict[str, Any]] = []
            for task_data in tasks.values():
                all_records.extend(list(task_data.get("records") or []))

            dimension_score = min(
                sum(int(task.get("score", 0) or 0) for task in tasks.values()),
                standard_max_score,
            )
            dimension_max_score = min(
                sum(int(task.get("max_score", 0) or 0) for task in tasks.values()),
                standard_max_score,
            )
            dimension_score = min(dimension_score, dimension_max_score) if dimension_max_score else 0

            scoring_data["dimensions"][dimension_id] = {
                **dimension_entry,
                "score": dimension_score,
                "max_score": dimension_max_score,
                "standard_max_score": standard_max_score,
                "question": question,
                "answer": answer,
                "evaluation_detail": evaluation_detail,
                "timestamp": timestamp,
                "records": all_records,
                "tasks": tasks,
            }

            self._recalculate_totals(scoring_data)
            total_score = scoring_data.get("total_score", 0)
            completed_max_score = scoring_data.get("completed_max_score", 0)
            
            # 保存到文件
            with open(scoring_file, 'w', encoding='utf-8') as f:
                json.dump(scoring_data, f, ensure_ascii=False, indent=2)
            
            saved_dim = self._normalize_dimension_entry(
                dimension_id,
                scoring_data["dimensions"].get(dimension_id, {})
            )

            try:
                if not db_save_mmse_score(
                    session_id,
                    dimension_id,
                    int(saved_dim.get('score', 0) or 0),
                    int(saved_dim.get('max_score', 0) or 0),
                    question,
                    answer,
                    evaluation_detail,
                ):
                    return json.dumps({
                        "success": False,
                        "message": "MMSE评分仅允许在认知筛查专项会话中记录",
                        "blocked_reason": "NON_COGNITIVE_SESSION",
                    }, ensure_ascii=False)
            except Exception as db_error:
                print(f"[MMSE] ⚠️ 同步 SQLite 失败: {db_error}")

            result = MMSEScoringResult(
                success=True,
                message=f"已记录 {dimension_id} 维度评分: {saved_dim.get('score', score)}/{saved_dim.get('max_score', max_score)}分",
                dimension_score=saved_dim.get('score', score),
                dimension_max_score=saved_dim.get('max_score', max_score),
                total_score=total_score,
                total_max_score=MMSE_TOTAL_SCORE,
                completed_max_score=completed_max_score,
                scaled_total_score=self._scale_to_reference_score(total_score, completed_max_score),
                coverage=round(completed_max_score / MMSE_TOTAL_SCORE, 3) if completed_max_score else 0.0,
                cognitive_status=scoring_data["cognitive_status"],
                completed_dimensions=list(scoring_data["dimensions"].keys())
            )
            
            print(
                f"[MMSE] ✅ 维度 {dimension_id} / 任务 {task_key}: "
                f"本次 {score}/{max_score} 分，锁定后维度 {saved_dim.get('score', 0)}/{saved_dim.get('max_score', 0)}，累计 {total_score}分"
            )
            
            return json.dumps(result.model_dump(), ensure_ascii=False)
            
        except Exception as e:
            return json.dumps({
                "success": False,
                "message": f"保存评分失败: {str(e)}"
            }, ensure_ascii=False)
    
    def _judge_cognitive_status(
        self,
        total_score: float,
        education_years: Optional[int] = None,
        education_level: Optional[str] = None,
    ) -> str:
        """根据教育水平对应正常界值解释30分口径MMSE分数。"""
        return self._build_cognitive_interpretation(total_score, education_years, education_level)["status"]
    
    def _get_score(self, scoring_data: Dict[str, Any], dimension_id: str) -> str:
        """获取指定维度的评分"""
        if dimension_id in scoring_data["dimensions"]:
            dim_data = self._normalize_dimension_entry(dimension_id, scoring_data["dimensions"][dimension_id])
            result = MMSEScoringResult(
                success=True,
                message=f"{dimension_id} 维度评分: {dim_data['score']}/{dim_data['max_score']}分",
                dimension_score=dim_data["score"],
                dimension_max_score=dim_data["max_score"],
                scoring_details=dim_data
            )
        else:
            result = MMSEScoringResult(
                success=False,
                message=f"未找到 {dimension_id} 维度的评分记录",
                dimension_score=0,
                dimension_max_score=MMSE_STANDARD_SCORES.get(dimension_id, 0)
            )
        
        return json.dumps(result.model_dump(), ensure_ascii=False)
    
    def _get_summary(
        self,
        scoring_data: Dict[str, Any],
        education_years: Optional[int] = None,
        education_level: Optional[str] = None,
    ) -> str:
        """获取MMSE总分和认知状态汇总"""
        total_score = scoring_data.get("total_score", 0)
        dimensions = scoring_data.get("dimensions", {})

        completed_max_score = scoring_data.get("completed_max_score")
        if completed_max_score is None:
            completed_max_score = sum(
                d.get("max_score", d.get("standard_max_score", 0)) for d in dimensions.values()
            )

        missing_dimensions = [
            dim for dim in MMSE_STANDARD_SCORES.keys() if dim not in dimensions
        ]

        scaled_total_score = self._scale_to_reference_score(total_score, completed_max_score)
        coverage = round(completed_max_score / MMSE_TOTAL_SCORE, 3) if completed_max_score else 0.0
        interpretation = self._build_cognitive_interpretation(scaled_total_score, education_years, education_level)
        if not total_score and not dimensions:
            cognitive_status = "未评估"
        elif completed_max_score == MMSE_TOTAL_SCORE:
            cognitive_status = interpretation["status"]
        else:
            cognitive_status = f"预估: {interpretation['status']} (覆盖率{completed_max_score}/{MMSE_TOTAL_SCORE})"
        
        # 统计各维度得分
        dimension_summary = {}
        for dim_id, dim_data in dimensions.items():
            dim_data = self._normalize_dimension_entry(dim_id, dim_data)
            dim_standard_max = dim_data.get("standard_max_score", MMSE_STANDARD_SCORES.get(dim_id, 0))
            dim_effective_max = dim_data.get("max_score", dim_standard_max)
            task_summary = {}
            task_items = []
            for task_id, task_data in sorted((dim_data.get("tasks") or {}).items(), key=lambda item: self._task_sort_key(item[0])):
                records = list(task_data.get("records") or [])
                latest_record = records[-1] if records else task_data
                task_score = int(task_data.get("score", 0) or 0)
                task_max = int(task_data.get("max_score", 0) or 0)
                base_task_id, _ = self._split_task_id(task_id)
                task_item = {
                    "task_id": task_id,
                    "base_task_id": base_task_id,
                    "label": self._format_task_label(task_id),
                    "score": task_score,
                    "max_score": task_max,
                    "status": self._score_status(task_score, task_max),
                    "question": str(latest_record.get("question", task_data.get("question", "")) or ""),
                    "answer": str(latest_record.get("answer", task_data.get("answer", "")) or ""),
                    "evaluation_detail": str(latest_record.get("evaluation_detail", task_data.get("evaluation_detail", "")) or ""),
                    "detail": self._clean_evaluation_detail(latest_record.get("evaluation_detail", task_data.get("evaluation_detail", ""))),
                    "timestamp": latest_record.get("timestamp", task_data.get("timestamp")),
                    "records_count": len(records),
                }
                task_summary[task_id] = task_item
                task_items.append(task_item)
            dimension_summary[dim_id] = {
                "score": dim_data["score"],
                "max_score": dim_effective_max,
                "standard_max_score": dim_standard_max,
                "task_count": len(dim_data.get("tasks") or {}),
                "percentage": round(dim_data["score"] / dim_effective_max * 100, 1) if dim_effective_max else 0.0,
                "tasks": task_summary,
                "items": task_items,
            }
        
        result = MMSEScoringResult(
            success=True,
            message=f"MMSE总分: {total_score}/{MMSE_TOTAL_SCORE}分 - {cognitive_status}",
            total_score=total_score,
            total_max_score=MMSE_TOTAL_SCORE,
            completed_max_score=completed_max_score,
            scaled_total_score=scaled_total_score,
            coverage=coverage,
            missing_dimensions=missing_dimensions,
            cognitive_status=cognitive_status,
            completed_dimensions=list(dimensions.keys()),
            scoring_details={
                "total_score": total_score,
                "total_max_score": MMSE_TOTAL_SCORE,
                "completed_max_score": completed_max_score,
                "scaled_total_score": scaled_total_score,
                "coverage": coverage,
                "missing_dimensions": missing_dimensions,
                "cognitive_status": cognitive_status,
                "education_level": interpretation["education_level"],
                "normal_threshold": interpretation["normal_threshold"],
                "score_basis": interpretation["score_basis"],
                "interpretation": interpretation,
                "dimension_scores": dimension_summary,
                "completed_dimensions": len(dimensions),
                "total_dimensions": 6,
                "completion_rate": round(len(dimensions) / 6 * 100, 1)
            }
        )
        
        return json.dumps(result.model_dump(), ensure_ascii=False)
    
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")
