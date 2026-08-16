from __future__ import annotations

from collections.abc import Callable
import json
from difflib import SequenceMatcher
from typing import Any, Dict, Optional

from .catalog import ScreeningTaskCatalog
from .state import ScreeningSessionState
from .task_planning import ScreeningTaskPlanning
from .tool_gateway import ScreeningToolGateway


class ScreeningAnswerEvaluation:
    """Evaluate task answers and persist their MMSE scores."""

    def __init__(
        self,
        *,
        state: ScreeningSessionState,
        catalog: ScreeningTaskCatalog,
        tools: ScreeningToolGateway,
        task_planner: ScreeningTaskPlanning,
        dimension_map: Dict[str, Dict[str, Any]],
        log_summary: Callable[[str, Dict[str, Any]], None],
    ) -> None:
        self.state = state
        self.catalog = catalog
        self.tool_gateway = tools
        self.task_planner = task_planner
        self.dimension_map = dimension_map
        self._log_summary = log_summary

    def _log_summary_card(
        self,
        title: str,
        items: Dict[str, Any],
    ) -> None:
        self._log_summary(title, items)

    def _try_rule_based_evaluation(
        self,
        task_id: str,
        user_answer: str,
        expected_answer: Optional[str],
    ) -> Optional[Dict]:
        """Evaluate deterministic task families without an LLM call."""
        if not task_id or task_id in self.catalog.buffer_tasks:
            return None
        answer = user_answer.strip()
        if not answer:
            return None
        for evaluator in (
            self._evaluate_attention_rule,
            self._evaluate_language_rule,
            self._evaluate_time_orientation_rule,
            self._evaluate_place_orientation_rule,
            self._evaluate_recall_rule,
        ):
            result = evaluator(task_id, answer, expected_answer)
            if result is not None:
                return result
        return None

    def _evaluate_attention_rule(
        self,
        task_id: str,
        answer: str,
        expected_answer: Optional[str],
    ) -> Optional[Dict]:
        if task_id == "attention_calc_life_math":
            import re
            numbers = [int(n) for n in re.findall(r'\d+', answer)]
            if numbers:
                expected_sequence = [93, 86, 79, 72, 65]
                if expected_answer is not None:
                    exp_nums = [int(n) for n in re.findall(r'\d+', str(expected_answer))]
                    if len(exp_nums) >= 5:
                        expected_sequence = exp_nums[:5]

                # ASR 有时会把题干里的 100、7 带入回答，先剔除明显题干数字。
                filtered_numbers = [n for n in numbers if n not in (100, 7)]
                user_numbers = (filtered_numbers or numbers)[:5]

                score = 0
                details = []
                previous_value = 100
                for idx, user_num in enumerate(user_numbers):
                    absolute_expected = expected_sequence[idx] if idx < len(expected_sequence) else previous_value - 7
                    relative_expected = previous_value - 7
                    is_step_correct = user_num == absolute_expected or user_num == relative_expected
                    if is_step_correct:
                        score += 1
                    details.append(f"第{idx + 1}步 {user_num}/期望{absolute_expected}{'✓' if is_step_correct else '×'}")
                    previous_value = user_num

                if score >= 5:
                    quality, performance = 'excellent', '正常'
                elif score >= 3:
                    quality, performance = 'good', '轻度异常'
                elif score >= 1:
                    quality, performance = 'fair', '中度异常'
                else:
                    quality, performance = 'poor', '重度异常'

                return {
                    'is_correct': score == 5,
                    'quality_level': quality,
                    'cognitive_performance': performance,
                    'is_complete': True,
                    'evaluation_detail': f"规则评估: 连续减7得分 {score}/5；患者回答 {user_numbers}；" + "，".join(details),
                    'need_followup': False,
                    'confidence': 1.0,
                    'raw_score': score,
                    'raw_max_score': 5,
                }
        if task_id == "attention_reverse_phrase" and expected_answer:
            answer_norm = self.task_planner._normalize_text(answer)
            expected_norm = self.task_planner._normalize_text(str(expected_answer))
            if expected_norm and answer_norm:
                def _lcs_len(text_a: str, text_b: str) -> int:
                    prev = [0] * (len(text_b) + 1)
                    for ch_a in text_a:
                        curr = [0]
                        for idx_b, ch_b in enumerate(text_b, 1):
                            if ch_a == ch_b:
                                curr.append(prev[idx_b - 1] + 1)
                            else:
                                curr.append(max(prev[idx_b], curr[-1]))
                        prev = curr
                    return prev[-1]

                raw_score = len(expected_norm) if expected_norm in answer_norm else min(len(expected_norm), _lcs_len(answer_norm, expected_norm))
                if raw_score >= len(expected_norm):
                    quality_level = 'excellent'
                    cognitive_performance = '正常'
                elif raw_score >= max(3, len(expected_norm) - 1):
                    quality_level = 'good'
                    cognitive_performance = '轻度异常'
                elif raw_score > 0:
                    quality_level = 'fair'
                    cognitive_performance = '中度异常'
                else:
                    quality_level = 'poor'
                    cognitive_performance = '异常'

                return {
                    'is_correct': raw_score == len(expected_norm),
                    'quality_level': quality_level,
                    'cognitive_performance': cognitive_performance,
                    'is_complete': True,
                    'evaluation_detail': f'规则评估: 倒序复述命中 {raw_score}/{len(expected_norm)}',
                    'need_followup': False,
                    'confidence': 0.95,
                    'raw_score': raw_score,
                    'raw_max_score': len(expected_norm),
                }
        return None

    def _evaluate_language_rule(
        self,
        task_id: str,
        answer: str,
        expected_answer: Optional[str],
    ) -> Optional[Dict]:
        if task_id == "language_naming_watch":
            if any(kw in answer for kw in ['手表', '表', '钟表', '腕表']):
                return {
                    'is_correct': True, 'quality_level': 'excellent',
                    'cognitive_performance': '正常', 'is_complete': True,
                    'evaluation_detail': '规则评估: 正确命名手表',
                    'need_followup': False, 'confidence': 1.0
                }
            elif len(answer) <= 10:  # 短回答但不含关键词 = 错误
                return {
                    'is_correct': False, 'quality_level': 'poor',
                    'cognitive_performance': '异常', 'is_complete': True,
                    'evaluation_detail': f'规则评估: 未命名手表 ({answer})',
                    'need_followup': False, 'confidence': 0.8
                }
        if task_id == "language_naming_pencil":
            if any(kw in answer for kw in ['铅笔', '笔', '钢笔', '圆珠笔']):
                return {
                    'is_correct': True, 'quality_level': 'excellent',
                    'cognitive_performance': '正常', 'is_complete': True,
                    'evaluation_detail': '规则评估: 正确命名铅笔',
                    'need_followup': False, 'confidence': 1.0
                }
            elif len(answer) <= 10:
                return {
                    'is_correct': False, 'quality_level': 'poor',
                    'cognitive_performance': '异常', 'is_complete': True,
                    'evaluation_detail': f'规则评估: 未命名铅笔 ({answer})',
                    'need_followup': False, 'confidence': 0.8
                }
        if task_id == "language_repetition_sentence" and expected_answer:
            expected_norm = self.task_planner._normalize_text(str(expected_answer))
            answer_norm = self.task_planner._normalize_text(answer)
            if expected_norm and answer_norm:
                ratio = SequenceMatcher(None, answer_norm, expected_norm).ratio()
                if (
                    answer_norm == expected_norm
                    or (len(answer_norm) >= 4 and answer_norm in expected_norm)
                    or (len(expected_norm) >= 4 and expected_norm in answer_norm)
                    or ratio >= 0.75
                ):
                    return {
                        'is_correct': True, 'quality_level': 'excellent',
                        'cognitive_performance': '正常', 'is_complete': True,
                        'evaluation_detail': f'规则评估: 复述句子命中 (ratio={ratio:.2f})',
                        'need_followup': False, 'confidence': 0.95
                    }
        return None

    def _evaluate_time_orientation_rule(
        self,
        task_id: str,
        answer: str,
        expected_answer: Optional[str],
    ) -> Optional[Dict]:
        if task_id == "orientation_time_year" and expected_answer:
            import re as _re
            exp_years = _re.findall(r'(\d{4})', str(expected_answer))
            user_years = _re.findall(r'(\d{4})', answer)
            # 也尝试匹配中文 "二零二六" 等
            cn_digits = {'零': '0', '〇': '0', '一': '1', '二': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8', '九': '9'}
            cn_year = ''
            for ch in answer:
                if ch in cn_digits:
                    cn_year += cn_digits[ch]
            if len(cn_year) == 4:
                user_years.append(cn_year)
            if exp_years and user_years:
                if user_years[0] == exp_years[0]:
                    return {
                        'is_correct': True, 'quality_level': 'excellent',
                        'cognitive_performance': '正常', 'is_complete': True,
                        'evaluation_detail': f'规则评估: 年份正确 ({user_years[0]})',
                        'need_followup': False, 'confidence': 1.0
                    }
                else:
                    return {
                        'is_correct': False, 'quality_level': 'poor',
                        'cognitive_performance': '异常', 'is_complete': True,
                        'evaluation_detail': f'规则评估: 年份不正确 (答{user_years[0]}, 期望{exp_years[0]})',
                        'need_followup': False, 'confidence': 0.9
                    }
        if task_id == "orientation_time_season" and expected_answer:
            season_keywords = {'春': '春', '夏': '夏', '秋': '秋', '冬': '冬'}
            expected_season = None
            for k, v in season_keywords.items():
                if k in str(expected_answer):
                    expected_season = v
                    break
            user_season = None
            for k, v in season_keywords.items():
                if k in answer:
                    user_season = v
                    break
            if expected_season and user_season:
                if user_season == expected_season:
                    return {
                        'is_correct': True, 'quality_level': 'excellent',
                        'cognitive_performance': '正常', 'is_complete': True,
                        'evaluation_detail': f'规则评估: 季节正确 ({user_season}季)',
                        'need_followup': False, 'confidence': 1.0
                    }
                else:
                    return {
                        'is_correct': False, 'quality_level': 'poor',
                        'cognitive_performance': '异常', 'is_complete': True,
                        'evaluation_detail': f'规则评估: 季节不正确 (答{user_season}, 期望{expected_season})',
                        'need_followup': False, 'confidence': 0.9
                    }
        if task_id == "orientation_time_month_date" and expected_answer:
            import re as _re
            # 提取期望月和日
            exp_m = _re.findall(r'(\d+)\s*月', str(expected_answer))
            exp_d = _re.findall(r'(\d+)\s*日', str(expected_answer))
            usr_m = _re.findall(r'(\d+)\s*月', answer)
            usr_d = _re.findall(r'(\d+)\s*[日号]', answer)
            correct_count = 0
            total = 0
            if exp_m:
                total += 1
                if usr_m and usr_m[0] == exp_m[0]:
                    correct_count += 1
            if exp_d:
                total += 1
                if usr_d and usr_d[0] == exp_d[0]:
                    correct_count += 1
            if total > 0:
                if correct_count == total:
                    return {
                        'is_correct': True, 'quality_level': 'excellent',
                        'cognitive_performance': '正常', 'is_complete': True,
                        'evaluation_detail': f'规则评估: 月份日期全部正确 ({correct_count}/{total})',
                        'need_followup': False, 'confidence': 1.0
                    }
                elif correct_count > 0:
                    return {
                        'is_correct': True, 'quality_level': 'fair',
                        'cognitive_performance': '轻度异常', 'is_complete': True,
                        'evaluation_detail': f'规则评估: 月份日期部分正确 ({correct_count}/{total})',
                        'need_followup': False, 'confidence': 0.85
                    }
                else:
                    return {
                        'is_correct': False, 'quality_level': 'poor',
                        'cognitive_performance': '异常', 'is_complete': True,
                        'evaluation_detail': f'规则评估: 月份日期均不正确',
                        'need_followup': False, 'confidence': 0.85
                    }
        if task_id == "orientation_time_weekday" and expected_answer:
            weekday_map = {
                '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '日': 7, '天': 7,
                '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
            }
            # 从 expected_answer 提取期望星期
            expected_day = None
            for k, v in weekday_map.items():
                if k in (expected_answer or ''):
                    expected_day = v
                    break
            # 从用户回答提取
            user_day = None
            for k, v in weekday_map.items():
                if k in answer:
                    user_day = v
                    break
            
            if expected_day and user_day:
                if user_day == expected_day:
                    return {
                        'is_correct': True, 'quality_level': 'excellent',
                        'cognitive_performance': '正常', 'is_complete': True,
                        'evaluation_detail': f'规则评估: 星期正确',
                        'need_followup': False, 'confidence': 1.0
                    }
                else:
                    return {
                        'is_correct': False, 'quality_level': 'poor',
                        'cognitive_performance': '异常', 'is_complete': True,
                        'evaluation_detail': f'规则评估: 星期不正确 (答{user_day}, 期望{expected_day})',
                        'need_followup': False, 'confidence': 0.9
                    }
        return None

    def _evaluate_place_orientation_rule(
        self,
        task_id: str,
        answer: str,
        expected_answer: Optional[str],
    ) -> Optional[Dict]:
        if task_id == "orientation_place_province_city" and expected_answer:
            exp_parts = [p.strip() for p in str(expected_answer).replace('，', ',').split(',') if p.strip()]
            correct_count = sum(1 for p in exp_parts if p and p in answer)
            total = len(exp_parts)
            if total > 0:
                if correct_count == total:
                    return {
                        'is_correct': True, 'quality_level': 'excellent',
                        'cognitive_performance': '正常', 'is_complete': True,
                        'evaluation_detail': f'规则评估: 省市全部正确 ({correct_count}/{total})',
                        'need_followup': False, 'confidence': 0.9
                    }
                elif correct_count > 0:
                    return {
                        'is_correct': True, 'quality_level': 'fair',
                        'cognitive_performance': '轻度异常', 'is_complete': True,
                        'evaluation_detail': f'规则评估: 省市部分正确 ({correct_count}/{total})',
                        'need_followup': False, 'confidence': 0.8
                    }
        if task_id == "orientation_place_district" and expected_answer:
            exp = str(expected_answer).strip()
            if exp and exp in answer:
                return {
                    'is_correct': True, 'quality_level': 'excellent',
                    'cognitive_performance': '正常', 'is_complete': True,
                    'evaluation_detail': f'规则评估: 区/县正确',
                    'need_followup': False, 'confidence': 0.9
                }
        if task_id == "orientation_place_location_floor" and expected_answer:
            import re as _re

            def _normalize_floor_value(text: str) -> str:
                raw = str(text or "")
                raw = raw.replace('层楼', '楼').replace('楼层', '楼').replace('层', '楼')
                match = _re.search(r'([0-9一二三四五六七八九十两]+)\s*楼', raw)
                if not match:
                    return ""
                value = match.group(1)
                digit_map = {'一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
                if value.isdigit():
                    return value
                if value in digit_map:
                    return str(digit_map[value])
                if value == '十':
                    return '10'
                if len(value) == 2 and value.startswith('十') and value[1] in digit_map:
                    return str(10 + digit_map[value[1]])
                if len(value) == 2 and value.endswith('十') and value[0] in digit_map:
                    return str(digit_map[value[0]] * 10)
                if len(value) == 3 and value[1] == '十' and value[0] in digit_map and value[2] in digit_map:
                    return str(digit_map[value[0]] * 10 + digit_map[value[2]])
                return value

            exp_parts = [p.strip() for p in str(expected_answer).replace('，', ',').split(',') if p.strip()]
            exp_place = exp_parts[0] if exp_parts else ""
            exp_floor = exp_parts[1] if len(exp_parts) > 1 else ""
            correct_count = 0
            total = 0

            if exp_place:
                total += 1
                place_tokens = [token.strip() for token in _re.split(r'[\s/、，,]+', exp_place) if token.strip() and len(token.strip()) >= 2]
                if exp_place in answer or any(token in answer for token in place_tokens):
                    correct_count += 1

            if exp_floor:
                total += 1
                exp_floor_value = _normalize_floor_value(exp_floor)
                ans_floor_value = _normalize_floor_value(answer)
                if exp_floor in answer or (exp_floor_value and ans_floor_value and exp_floor_value == ans_floor_value):
                    correct_count += 1

            if total > 0:
                if correct_count == total:
                    return {
                        'is_correct': True, 'quality_level': 'excellent',
                        'cognitive_performance': '正常', 'is_complete': True,
                        'evaluation_detail': f'规则评估: 手填地址地点和楼层全部正确 ({correct_count}/{total})',
                        'need_followup': False, 'confidence': 0.9
                    }
                elif correct_count > 0:
                    return {
                        'is_correct': True, 'quality_level': 'fair',
                        'cognitive_performance': '轻度异常', 'is_complete': True,
                        'evaluation_detail': f'规则评估: 手填地址地点或楼层部分正确 ({correct_count}/{total})',
                        'need_followup': False, 'confidence': 0.8
                    }
                else:
                    return {
                        'is_correct': False, 'quality_level': 'poor',
                        'cognitive_performance': '异常', 'is_complete': True,
                        'evaluation_detail': '规则评估: 手填地址地点和楼层均不正确',
                        'need_followup': False, 'confidence': 0.8
                    }
        return None

    def _evaluate_recall_rule(
        self,
        task_id: str,
        answer: str,
        expected_answer: Optional[str],
    ) -> Optional[Dict]:
        if task_id == "recall_3words":
            memory_words = self.state.session_data.get('memory_words')
            if memory_words and isinstance(memory_words, list):
                correct_count = sum(1 for w in memory_words if w in answer)
                if correct_count == len(memory_words):
                    quality = 'excellent'
                elif correct_count >= 2:
                    quality = 'good'
                elif correct_count >= 1:
                    quality = 'fair'
                else:
                    quality = 'poor'
                return {
                    'is_correct': correct_count > 0, 'quality_level': quality,
                    'cognitive_performance': '正常' if correct_count >= 2 else '轻度异常' if correct_count == 1 else '异常',
                    'is_complete': True,
                    'evaluation_detail': f'规则评估: 记住{correct_count}/{len(memory_words)}个词',
                    'need_followup': False, 'confidence': 0.9
                }
        return None

    def _get_expected_answer_for_task(self, task_id: str, patient_profile: Dict) -> Optional[str]:
        """获取任务的期望答案（用于精准评估）"""
        from src.utils.location_service import get_realtime_context
        
        if not task_id or task_id in self.catalog.buffer_tasks:
            return None
        
        ctx = get_realtime_context()
        time_info = ctx.get('time', {})
        location = ctx.get('location', {})
        
        if task_id == "orientation_time_year":
            return f"{time_info.get('year', '未知')}年"
        elif task_id == "orientation_time_season":
            return f"{time_info.get('season', '未知')}"
        elif task_id == "orientation_time_month_date":
            return f"{time_info.get('month')}月{time_info.get('day')}日"
        elif task_id == "orientation_time_weekday":
            return f"星期{time_info.get('weekday', '未知')}"
        elif task_id == "orientation_place_province_city":
            return f"{location.get('province', '')}，{location.get('city', '')}"
        elif task_id == "orientation_place_district":
            return f"{location.get('district', '')}"
        elif task_id == "orientation_place_location_floor":
            place_parts = []
            for part in [location.get('place', ''), location.get('hospital', ''), location.get('department', ''), location.get('bed_number', '')]:
                part = str(part or '').strip()
                if part and part not in place_parts:
                    place_parts.append(part)
            place_text = ' '.join(place_parts).strip()
            floor_text = str(location.get('floor', '') or '').strip()
            if place_text and floor_text:
                return f"{place_text}，{floor_text}"
            if place_text:
                return place_text
            return floor_text or None
        elif task_id == "registration_3words":
            words = self.state.session_data.get('memory_words')
            return f"记住三个词：{words}" if words else None
        elif task_id == "recall_3words":
            words = self.state.session_data.get('memory_words')
            return f"三个词是：{words}" if words else None
        elif task_id == "attention_calc_life_math":
            calc_cfg = self.state.session_data.get('calculation_config') or {}
            expected_answer = calc_cfg.get('expected_answer') or calc_cfg.get('expected_sequence')
            return str(expected_answer) if expected_answer is not None else "93、86、79、72、65"
        elif task_id == "attention_reverse_phrase":
            return "安平入出祝"
        elif task_id == "language_naming_watch":
            return "手表"
        elif task_id == "language_naming_pencil":
            return "铅笔"
        elif task_id == "language_repetition_sentence":
            return "非如果，还有，或但是"
        elif task_id == "language_reading_close_eyes":
            return "闭上眼睛"
        elif task_id == "language_3step_action":
            return "举起右手、握拳、把手放到胸前"
        
        return None
    def _try_consume_pending_task_answer(
        self,
        pending_task_id: str,
        user_answer: str,
        patient_profile: Dict,
        session_id: str,
        fallback_dimension_id: str,
        fallback_dimension_name: str,
    ) -> Optional[Dict[str, Any]]:
        """如果用户已经直接回答了 pending task，则直接消费并记分。"""
        if pending_task_id != "language_repetition_sentence":
            return None

        expected_answer = self._get_expected_answer_for_task(pending_task_id, patient_profile)
        normalized_answer = self.task_planner._normalize_text(user_answer)
        normalized_expected = self.task_planner._normalize_text(expected_answer or "")
        if not normalized_answer or not normalized_expected:
            return None

        similarity = SequenceMatcher(None, normalized_answer, normalized_expected).ratio()
        if not (
            normalized_answer == normalized_expected
            or (len(normalized_answer) >= 4 and normalized_answer in normalized_expected)
            or (len(normalized_expected) >= 4 and normalized_expected in normalized_answer)
            or similarity >= 0.70
        ):
            return None

        pending_question = f"请跟着我重复一遍：\u300c{expected_answer}\u300d" if expected_answer else "请跟着我重复一遍。"
        eval_result = self._try_rule_based_evaluation(pending_task_id, user_answer, expected_answer)
        if eval_result is None:
            eval_result = self.tool_gateway._call_answer_evaluation(
                pending_question,
                user_answer,
                pending_task_id,
                patient_profile,
                expected_answer,
            )

        task_cfg = self.catalog.task_config.get(pending_task_id, {})
        pending_dim_id = task_cfg.get('dimension_id', fallback_dimension_id)
        pending_dim_name = self.dimension_map.get(pending_dim_id, {}).get('name', fallback_dimension_name)

        current_turns = self.state._task_turns.get(pending_task_id, 0) + 1
        self.state._task_turns[pending_task_id] = current_turns

        mmse_result = self._call_score_recording(
            session_id,
            pending_dim_id,
            eval_result,
            pending_question,
            user_answer,
            task_cfg.get('max_points'),
            task_id=pending_task_id,
            expected_answer=expected_answer,
            current_turn=current_turns,
        )

        min_turns = task_cfg.get('min_turns', 1)
        if current_turns >= min_turns:
            self.state._task_done.add(pending_task_id)
            cognitive_done = [t for t in self.state._task_done if t not in self.catalog.buffer_tasks]
            self._log_summary_card(
                "Task Progress",
                {
                    "task": pending_task_id,
                    "status": "completed_via_pending_answer",
                    "turns": f"{current_turns}/{min_turns}",
                    "buffer": False,
                    "cognitive_done": len(cognitive_done),
                },
            )

        self.state._pending_consent_task_id = None
        if pending_task_id == self.state._consent_granted_task_id:
            self.state._consent_granted_task_id = None

        return {
            'task_id': pending_task_id,
            'dimension_id': pending_dim_id,
            'dimension_name': pending_dim_name,
            'eval_result': eval_result,
            'mmse_result': mmse_result,
        }
    def _call_score_recording(
        self, session_id: str, dimension_id: str, eval_result: Dict,
        question: str, answer: str, max_score_override: Optional[int] = None,
        task_id: Optional[str] = None, expected_answer: Optional[str] = None,
        current_turn: Optional[int] = None,
    ) -> Dict:
        """调用评分记录工具（定性 + MMSE定量），返回MMSE评分信息"""
        # 1. 定性评估记录
        self.tool_gateway.score_tool._run(
            session_id=session_id,
            dimension_id=dimension_id,
            quality_level=eval_result.get('quality_level', 'fair'),
            cognitive_performance=eval_result.get('cognitive_performance', '正常'),
            question=question,
            answer=answer,
            evaluation_detail=eval_result.get('evaluation_detail', ''),
            action='save'
        )

        mmse_task_id = self._build_mmse_task_id(task_id, expected_answer, current_turn)
        effective_max_score = self._get_mmse_task_max_score(task_id, max_score_override)
        
        # 2. ⭐ MMSE定量评分（根据质量等级估算分数）
        if eval_result.get('raw_score') is not None:
            raw_max_score = int(eval_result.get('raw_max_score', effective_max_score or max_score_override or 0) or 0)
            final_max_score = int(effective_max_score if effective_max_score is not None else raw_max_score)
            if raw_max_score > 0:
                final_max_score = raw_max_score if final_max_score <= 0 else min(max(final_max_score, raw_max_score), raw_max_score)
            raw_score = int(eval_result.get('raw_score', 0) or 0)
            raw_score = max(0, min(raw_score, final_max_score)) if final_max_score > 0 else 0
            mmse_score = {
                'score': raw_score,
                'max_score': final_max_score,
            }
        else:
            mmse_score = self._convert_quality_to_mmse_score(
                dimension_id,
                eval_result.get('quality_level', 'fair'),
                eval_result.get('cognitive_performance', '正常'),
                max_score_override=effective_max_score,
            )
        
        mmse_result = {
            'dimension_id': dimension_id,
            'score': mmse_score['score'],
            'max_score': mmse_score['max_score'],
            'total_score': 0
        }
        
        try:
            result_json = self.tool_gateway.mmse_tool._run(
                session_id=session_id,
                dimension_id=dimension_id,
                score=mmse_score['score'],
                max_score=mmse_score['max_score'],
                task_id=mmse_task_id,
                question=question,
                answer=answer,
                evaluation_detail=f"质量等级: {eval_result.get('quality_level')} | {eval_result.get('evaluation_detail', '')}",
                action='save'
            )
            self._log_summary_card(
                "MMSE Score",
                {
                    "dimension": dimension_id,
                    "task": mmse_task_id or task_id or dimension_id,
                    "score": f"{mmse_score['score']}/{mmse_score['max_score']}",
                },
            )
            
            # 从save的返回结果中获取总分
            save_result = json.loads(result_json)
            mmse_result['total_score'] = save_result.get('total_score', 0)
            
        except Exception as e:
            self._log_summary_card("MMSE Score", {"status": "failed", "error": str(e)})
        
        return mmse_result
    def _build_mmse_task_id(
        self,
        task_id: Optional[str],
        expected_answer: Optional[str] = None,
        current_turn: Optional[int] = None,
    ) -> Optional[str]:
        if not task_id:
            return None

        if task_id == "attention_calc_life_math":
            return task_id

        return task_id
    def _get_mmse_task_max_score(self, task_id: Optional[str], max_score_override: Optional[int]) -> Optional[int]:
        return max_score_override
    def _convert_quality_to_mmse_score(
        self,
        dimension_id: str,
        quality_level: str,
        cognitive_performance: str,
        max_score_override: Optional[int] = None,
    ) -> Dict:
        """
        将质量等级转换为MMSE分数
        
        映射逻辑：
        1. LLM按维度生成灵活问题（例如"您知道今天几号吗？"）
        2. AnswerEvaluationTool评估回答质量（excellent/good/fair/poor）
        3. 根据维度和质量等级映射到MMSE标准分数
        
        映射规则：
        - excellent: 完全正确，满分或接近满分
        - good: 基本正确，80%左右
        - fair: 部分正确或不确定，60%左右
        - poor: 明显错误或无法回答，30%或更低
        
        注：虽然问法灵活，但评估的认知能力点是固定的，所以可以映射到标准分数
        """
        # MMSE标准分值（国际通用）
        max_scores = {
            "orientation": 10,          # 时间5分+地点5分
            "registration": 3,          # 三个词即时记忆
            "attention_calculation": 10, # 连续减7(5分) + 倒着说词组(5分)
            "recall": 3,                # 延迟回忆三个词
            "language": 8,              # 命名、复述、指令、阅读、书写
            "copy": 1                   # 临摹五边形
        }
        
        max_score = int(max_score_override) if max_score_override is not None else max_scores.get(dimension_id, 0)
        
        # 🎯 优化后的映射规则（更符合实际评估）
        if quality_level == "excellent":
            # 回答完全正确、清晰准确
            score = max_score  # 给满分
        elif quality_level == "good":
            # 回答基本正确，但可能有小瑕疵
            score = max(1, int(max_score * 0.80))  # 80%，至少1分
        elif quality_level == "fair":
            # 回答部分正确或不确定
            score = max(1, int(max_score * 0.50))  # 50%，至少1分
        else:  # poor
            # 回答明显错误或无法回答
            score = int(max_score * 0.20)  # 20%，可能是0分
        
        # 🔧 根据认知表现进一步微调（避免高估）
        if cognitive_performance == "重度异常":
            score = min(score, int(max_score * 0.3))
        elif cognitive_performance == "中度异常":
            score = min(score, int(max_score * 0.6))
        elif cognitive_performance == "轻度异常":
            score = min(score, int(max_score * 0.8))
        
        return {"score": score, "max_score": max_score}
    def _calculate_alzheimers_risk(self, total_score: int) -> Dict:
        """
        根据MMSE总分计算阿尔茨海默病风险
        
        MMSE分数解释（国际标准）：
        - 24-30分：认知功能正常
        - 18-23分：轻度认知障碍
        - 10-17分：中度痴呆
        - 0-9分：重度痴呆
        
        Returns:
            包含 risk_level, probability, description, recommendation 的字典
        """
        if total_score >= 24:
            return {
                'risk_level': '低风险',
                'probability': '低于10%',
                'score_range': '24-30分',
                'description': '认知功能正常，阿尔茨海默病风险较低',
                'recommendation': '建议保持良好的生活习惯，定期进行认知筛查（每1-2年）',
                'severity': 'normal'
            }
        elif total_score >= 18:
            return {
                'risk_level': '中度风险',
                'probability': '30-50%',
                'score_range': '18-23分',
                'description': '轻度认知障碍，存在转变为阿尔茨海默病的风险',
                'recommendation': '强烈建议到医院神经内科进行进一步检查，包括头部MRI、血液检查等',
                'severity': 'mild'
            }
        elif total_score >= 10:
            return {
                'risk_level': '高风险',
                'probability': '60-80%',
                'score_range': '10-17分',
                'description': '中度认知障碍，高度怀疑阿尔茨海默病或其他痴呆',
                'recommendation': '需要立即就医，进行全面的神经系统检查和评估，尽早干预治疗',
                'severity': 'moderate'
            }
        else:  # 0-9
            return {
                'risk_level': '极高风险',
                'probability': '高于85%',
                'score_range': '0-9分',
                'description': '重度认知障碍，极有可能已患阿尔茨海默病或其他重度痴呆',
                'recommendation': '紧急就医！需要立即到三甲医院神经内科或记忆门诊就诊，开始专业治疗和护理',
                'severity': 'severe'
            }
