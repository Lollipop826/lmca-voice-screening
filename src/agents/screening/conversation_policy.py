from __future__ import annotations

from collections.abc import Callable
import os
import json
import time
import re
from typing import Any, Dict, List, Optional

from .catalog import ScreeningTaskCatalog
from .state import ScreeningSessionState
from .task_planning import ScreeningTaskPlanning
from .tool_gateway import ScreeningToolGateway


class ScreeningConversationPolicy:
    """Conversation-level retry, consent, comfort and completion policy."""

    def __init__(
        self,
        *,
        use_local: bool,
        state: ScreeningSessionState,
        catalog: ScreeningTaskCatalog,
        tools: ScreeningToolGateway,
        task_planner: ScreeningTaskPlanning,
        log_summary: Callable[[str, Dict[str, Any]], None],
        log_verbose: Callable[[str], None],
    ) -> None:
        self.use_local = use_local
        self.state = state
        self.catalog = catalog
        self.tool_gateway = tools
        self.task_planner = task_planner
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

    def _max_invalid_attempts(self) -> int:
        try:
            return max(1, int(os.getenv("MAX_INVALID_TASK_ATTEMPTS", "2")))
        except ValueError:
            return 2
    def _normalize_invalid_answer(self, text: str) -> str:
        return re.sub(r"\s+", "", (text or "").strip())
    def _detect_unable_answer(self, answer: str) -> bool:
        norm = self._normalize_invalid_answer(answer)
        if not norm:
            return True
        unable_patterns = [
            r"不知道", r"不记得", r"记不清", r"想不起来", r"想不起", r"忘了",
            r"不会", r"答不上来", r"说不上来", r"不清楚", r"不晓得", r"没印象",
        ]
        return any(re.search(pattern, norm) for pattern in unable_patterns)
    def _detect_refusal_answer(self, answer: str, resistance_result: Optional[Dict[str, Any]] = None) -> bool:
        norm = self._normalize_invalid_answer(answer)
        refusal_patterns = [
            r"不想回答", r"不回答", r"不想说", r"不说(了)?", r"不想做", r"不做(了|这个)?",
            r"别问(了)?", r"不聊(了)?", r"拒绝", r"不愿意", r"不配合", r"不想测", r"不测了",
        ]
        if any(re.search(pattern, norm) for pattern in refusal_patterns):
            return True
        if not resistance_result:
            return False
        return bool(resistance_result.get("is_resistant")) and resistance_result.get("turn_intent") == "refusal"
    def _invalid_answer_kind(
        self,
        answer: str,
        resistance_result: Optional[Dict[str, Any]] = None,
        eval_result: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if resistance_result and resistance_result.get("wants_repeat") and not resistance_result.get("has_substantive_answer"):
            return "repeat_request"
        if self._detect_refusal_answer(answer, resistance_result):
            return "refusal"
        if self._detect_unable_answer(answer):
            return "unable"
        if eval_result and eval_result.get("quality_level") == "poor" and not resistance_result:
            return "unable"
        return None
    def _register_invalid_attempt(self, task_id: Optional[str], kind: Optional[str]) -> int:
        if not task_id or not kind or task_id in self.catalog.buffer_tasks:
            return 0
        key = f"{task_id}:{kind}"
        self.state._task_attempts[key] = self.state._task_attempts.get(key, 0) + 1
        return self.state._task_attempts[key]
    def _clear_invalid_attempts(self, task_id: Optional[str]) -> None:
        if not task_id:
            return
        for key in list(self.state._task_attempts.keys()):
            if key == task_id or key.startswith(f"{task_id}:"):
                self.state._task_attempts.pop(key, None)
    def _build_retry_question(self, kind: str, doctor_question: str, attempt: int) -> str:
        if kind == "repeat_request":
            prefix = "没关系，我再说一遍：" if attempt <= 1 else "我换个简单说法："
        elif kind == "refusal":
            prefix = "没关系，这一题我们先放一放。等会儿您状态好些了，我们再回来看看："
        else:
            prefix = "没关系，想不起来也正常。您再试试看："
        return f"{prefix}{doctor_question}"
    def _enter_comfort_after_refusal(
        self,
        task_id: Optional[str],
        patient_answer: str,
        patient_profile: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        dimension_name: str,
        dimension_id: str,
        start_time: float,
    ) -> Dict[str, Any]:
        self.state.is_in_comfort_mode = True
        self.state.comfort_turn_count = 0
        self.state._comfort_entry_category = "refusal"
        if task_id and task_id not in self.catalog.buffer_tasks:
            self.state._comfort_interrupted_task_id = task_id
        self.state._last_task_id = "buffer_chat"

        try:
            comfort_result = self.tool_gateway.comfort_tool._run(
                resistance_category="refusal",
                patient_answer=patient_answer,
                patient_name=patient_profile.get('name'),
                patient_age=patient_profile.get('age'),
                patient_gender=patient_profile.get('gender'),
                used_topics=self.state._used_chat_topics,
                chat_history=chat_history,
                use_template=False,
            )
            comfort_data = json.loads(comfort_result)
            if topic := comfort_data.get('selected_topic'):
                self.state._used_chat_topics.append(topic)
                self.task_planner._set_bridge_context(
                    topic,
                    topic,
                    remember_topic=False,
                )
            comfort_message = comfort_data.get('comfort_message') or "没关系，这一题咱们先放一放，咱们轻松聊两句。"
        except Exception as exc:
            self._log_verbose(f"拒答安抚生成失败: {exc}")
            comfort_message = "没关系，这一题咱们先放一放，您不用着急，咱们轻松聊两句。"

        self.state._last_generated_question = comfort_message
        self._log_summary_card(
            "Resistance Deferral",
            {"task": task_id or "-", "action": "comfort_then_resume_later"},
        )
        return {
            'output': comfort_message,
            'response': comfort_message,
            'has_resistance': True,
            'resistance_category': 'refusal',
            'is_comfort_mode': True,
            'deferred_task_id': task_id,
            'dimension': dimension_name,
            'dimension_id': dimension_id,
            'total_time': time.time() - start_time
        }
    def _build_zero_score_eval(self, task_id: str, kind: str, answer: str, max_score: int) -> Dict[str, Any]:
        kind_zh = {
            "repeat_request": "多次请求重复，未形成可评分回答",
            "refusal": "多次拒绝回答",
            "unable": "多次表示答不上来",
        }.get(kind, "多次无有效回答")
        return {
            "is_correct": False,
            "quality_level": "poor",
            "cognitive_performance": "异常",
            "is_complete": True,
            "evaluation_detail": f"自动跳过并计0分: {kind_zh}; 患者回答: {answer}",
            "need_followup": False,
            "confidence": 1.0,
            "raw_score": 0,
            "raw_max_score": int(max_score or 0),
            "skipped": True,
            "skip_reason": kind,
        }
    @staticmethod
    def _get_full_name(patient_profile: Dict) -> str:
        """根据患者信息生成带称呼后缀的完整姓名，如'孙光飞先生'"""
        name = patient_profile.get('name', '')
        if not name:
            return ''
        gender = patient_profile.get('gender') or patient_profile.get('sex') or ''
        if gender == '男':
            suffix = '先生'
        elif gender == '女':
            suffix = '女士'
        else:
            suffix = ''
        return f"{name}{suffix}"
    def _get_recent_cognitive_task_id(self, last_task_id: Optional[str] = None) -> Optional[str]:
        if last_task_id and last_task_id not in self.catalog.buffer_tasks:
            return last_task_id
        for candidate in [
            self.state._last_cognitive_task_id,
            self.state._buffer_resume_task_id,
        ]:
            if candidate and candidate not in self.catalog.buffer_tasks:
                return candidate
        return None
    def _build_quick_ack(self, user_answer: Optional[str]) -> str:
        text = (user_answer or '').strip().rstrip('。！？!?~～')
        if not text:
            return ''
        normalized = self.task_planner._normalize_text(text)
        if any(kw in normalized for kw in ["有时候", "有时", "偶尔", "看情况"]):
            return "好的，我了解了。"
        if any(kw in normalized for kw in ["不会", "不常", "很少", "不大"]):
            return "没关系，我了解了。"
        if any(kw in normalized for kw in ["好啊", "好呀", "好的", "行啊", "行呀", "可以", "愿意", "试试", "会啊", "会呢", "会的", "没问题"]):
            return "好的。"
        if normalized in {"好", "行", "嗯", "会"}:
            return "好的。"
        if len(text) <= 8:
            return "好的。"
        return ''
    def _is_substantive_comfort_reply(self, user_answer: Optional[str]) -> bool:
        normalized = self.task_planner._normalize_text(user_answer or "")
        if not normalized:
            return False
        if any(kw in normalized for kw in ["结束", "再见", "拜拜", "不聊了", "就这样", "到这儿", "先这样", "回头聊"]):
            return False
        if any(kw in normalized for kw in ["不想", "不说", "别问", "不要", "不聊", "没完", "烦", "滚", "算了", "不做"]):
            return False
        if normalized in {"好", "好的", "嗯", "恩", "行", "可以", "是", "对", "不知道", "不记得", "忘了", "没啥", "没什么", "随便", "都行"}:
            return False
        return len(normalized) >= 2
    def _get_comfort_resume_min_turns(self) -> int:
        category = self.state._comfort_entry_category or "normal"
        if category in {"hostility", "distress"}:
            return 2
        return 1
    def _build_consent_prompt(self, task_id: str, patient_profile: Dict, user_answer: str = None) -> str:
        """用 LLM 生成自然的征求同意话语，包含对用户回答的反馈"""
        patient_name = self._get_full_name(patient_profile)
        quick_ack = self._build_quick_ack(user_answer)
        
        purpose = "玩个30秒小互动"
        if task_id == "attention_calc_life_math":
            purpose = "做个30秒小算术"
        elif task_id == "attention_reverse_phrase":
            purpose = "把我念的几个字倒着说出来"
        elif task_id in {"language_naming_watch", "language_naming_pencil"}:
            purpose = "看张图说说是什么"
        elif task_id == "language_repetition_sentence":
            purpose = "跟着我复述一句固定句子"
        elif task_id == "language_reading_close_eyes":
            purpose = "看一句话照着做一下，不用念出来"
        elif task_id == "language_3step_action":
            purpose = "按我说的做三个手部动作"

        # 🆕 改进方案：LLM 只生成回应，代码保证任务邀请
        try:
            import random
            if self.use_local:
                from src.llm.model_pool import get_pooled_llm
                llm = get_pooled_llm(pool_key='7b_complex')
            else:
                from src.llm.http_client_pool import get_chat_openai
                llm = get_chat_openai(temperature=0.7, max_tokens=60, timeout=10, max_retries=1)
            
            # 任务邀请模板（代码保证 purpose 出现）
            invite_templates = [
                f"诶，能陪我{purpose}不？不想来也行～",
                f"对了，咱们{purpose}呗？不难的～",
                f"您能帮我个忙不？就{purpose}～",
                f"要不咱来{purpose}？不想玩就算～",
            ]
            invite = random.choice(invite_templates)
            
            if user_answer and user_answer.strip():
                ack = quick_ack if len(user_answer.strip()) <= 8 and quick_ack else ""
                if not ack:
                    prompt = f"""你是{patient_name or '老人'}的晚辈，正在陪他/她聊天。
老人刚才说了：「{user_answer}」

请用1句简短的话回应老人说的内容（5-15字），要求：
- 针对老人说的具体内容回应
- 不要用万金油的「挺好」「不错」
- 口语化、接地气

示例：
- 老人说看书 → "余华的书确实好看！"
- 老人说位置 → "海淀区那边挺方便的"
- 老人说冷 → "是挺冷的，多穿点"

直接输出回应，不要加引号："""
                    response = llm.invoke([{"role": "user", "content": prompt}])
                    ack = response.content.strip() if hasattr(response, 'content') else str(response).strip()
                    ack = ack.strip('"').strip("'").strip('「').strip('」')
                
                if ack and len(ack) < 30:
                    result = f"{ack}，{invite}" if not ack.endswith(("！", "!", "。", ".")) else f"{ack}{invite}"
                    self._log_verbose(f"生成征求同意: ack='{ack}', invite='{invite}'")
                    return result
            
            if patient_name:
                result = f"{patient_name}，{invite}"
            else:
                result = invite
            self._log_verbose(f"使用模板征求同意: {result}")
            return result
        except Exception as e:
            self._log_summary_card("Consent Copy", {"status": "fallback_template", "error": str(e)})
        
        import random
        if user_answer and user_answer.strip():
            if quick_ack:
                templates = [
                    f"{quick_ack}来，咱们{purpose}呗？不想玩就算～",
                    f"{quick_ack}诶，能陪我{purpose}不？",
                    f"{quick_ack}对了，咱来{purpose}，您看行不？",
                ]
            else:
                templates = [
                    f"好嘞！{patient_name}，来，咱们{purpose}呗？不想玩就算～",
                    f"嗯嗯！诶{patient_name}，能陪我{purpose}不？",
                    f"说得好！对了{patient_name}，咱来{purpose}，您看行不？",
                ]
        else:
            templates = [
                f"{patient_name}，来，咱们玩个小游戏呗？不想玩就算～",
                f"诶{patient_name}，能陪我{purpose}不？就当逗我玩儿～",
                f"{patient_name}，我想跟您{purpose}，您看行不？不勉强哈～",
            ]
        return random.choice(templates)
    def _is_user_asking_question(self, user_answer: str) -> bool:
        """🔥 规则化检测用户是否在提问（替代 LLM，省 1-2 秒）"""
        if not user_answer or len(user_answer.strip()) < 4:
            return False
        
        text = user_answer.strip()

        # 排除：抱怨/反问式（不算真正提问）
        complaint_patterns = [
            '行不行', '好不好', '能不能别', '烦不烦', '有完没完',
            '不行吗', '不好吗', '可以吗', '算了吧', '别问', '不想',
            '我不知道', '忘了', '不记得', '没听清',
        ]
        if any(p in text for p in complaint_patterns):
            return False

        # 短句且无问号 → 不是提问
        if len(text) <= 5 and '？' not in text and '?' not in text:
            return False

        # 信息性提问关键短语（直接命中）
        info_phrases = ['天气怎么', '你叫什么', '你是谁', '现在几点', '能告诉我', '请问']
        if any(p in text for p in info_phrases):
            self._log_verbose(f"用户提问检测(规则): '{text[:30]}' → 是提问")
            return True

        # 疑问词 + 问号 → 大概率是提问
        question_words = ['什么', '怎么', '哪里', '哪个', '谁', '几点', '几号', '多少', '为什么', '为啥', '啥时候']
        has_qword = any(qw in text for qw in question_words)
        has_qmark = '？' in text or '?' in text

        if has_qword and has_qmark:
            self._log_verbose(f"用户提问检测(规则): '{text[:30]}' → 是提问")
            return True

        return False
    def _generate_answer_to_user_question(self, user_question: str, patient_profile: Dict, chat_history: List) -> str:
        """生成对用户问题的回答"""
        patient_name = patient_profile.get('name', '')
        greeting = f"{patient_name}，" if patient_name else ""
        
        # 🔥 传入结构化对话历史（而非字符串）
        recent_history = None
        if chat_history and len(chat_history) > 0:
            recent_history = chat_history[-6:]
        
        # 用LLM生成回答
        result_json = self.tool_gateway.question_tool._run(
            dimension_name="闲聊",
            dimension_description="回答对方的问题，然后自然地继续聊天",
            patient_name=patient_profile.get('name'),
            patient_gender=patient_profile.get('gender'),
            patient_age=patient_profile.get('age'),
            conversation_history=recent_history,
            task_instruction=f"对方问了一个问题：'{user_question}'。请先简短回答这个问题，然后自然地继续聊天。不要反问同样的问题。",
        )
        
        try:
            result = json.loads(result_json)
            if result.get('success') and result.get('question'):
                return result['question']
        except:
            pass
        
        # 兜底回答
        return f"{greeting}这个我也不太清楚呢，您怎么看？"
    def _generate_completion_message(
        self, 
        total_score: int, 
        risk_assessment: Dict, 
        patient_profile: Dict
    ) -> str:
        """
        生成评估完成后的温馨消息
        
        Args:
            total_score: MMSE总分
            risk_assessment: 风险评估结果
            patient_profile: 患者信息
            
        Returns:
            温馨的完成消息
        """
        patient_name = patient_profile.get('name', '')
        greeting = f"{patient_name}，" if patient_name else ""
        
        return (
            f"{greeting}今天聊得挺开心的，辛苦您啦！\n\n"
            f"咱们先歇一歇，您喝口水、活动活动。下次我再来跟您唠唠家常～"
        )
    def _check_user_willing_to_continue(self, user_input: str) -> bool:
        """
        检查用户是否表示愿意继续评估（规则快速路径 + LLM 兜底）
        
        策略（参考豆包等对话系统）：
        1. 明确同意/拒绝 → 规则秒回（0ms）
        2. 模糊/追问 → LLM 意图判断（~200-400ms）
        3. LLM 失败 → 默认视为同意
        """
        text = user_input.strip()
        if not text:
            self._log_verbose("同意检测(规则): 空输入 → 默认同意")
            return True
        
        # ===== 规则快速路径 =====
        # 排除伪否定
        false_neg = ["不错", "不好意思", "不客气", "不过", "不然"]
        cleaned = text
        for fp in false_neg:
            cleaned = cleaned.replace(fp, "")
        
        # 明确同意（秒回）
        positive = ["好", "行", "可以", "愿意", "试试", "继续", "开始",
                     "没问题", "OK", "ok", "嗯", "是的", "来吧", "说吧"]
        for kw in positive:
            if kw in text:
                has_neg = any(n in cleaned for n in ["不想", "不愿", "不行", "不要", "别"])
                if not has_neg:
                    self._log_verbose(f"同意检测(规则): 匹配 '{kw}' → 同意")
                    return True
        
        # 明确拒绝（秒回）
        negative = ["不想", "不愿", "不行", "不要", "不做", "不用", "不了",
                     "算了", "累了", "休息", "等会", "以后", "停", "够了"]
        for kw in negative:
            if kw in cleaned:
                self._log_verbose(f"同意检测(规则): 匹配 '{kw}' → 拒绝")
                return False
        
        # ===== LLM 兜底：模糊/追问/闲聊 =====
        self._log_verbose("同意检测(LLM): 规则未命中，调用LLM判断意图")
        return self._llm_check_consent_intent(text)
    def _llm_check_consent_intent(self, user_input: str) -> bool:
        """用快速小模型判断用户意图：同意/拒绝/追问"""
        try:
            t0 = time.time()
            if self.use_local:
                from src.llm.model_pool import get_pooled_llm
                llm = get_pooled_llm(pool_key='7b_complex')
            else:
                from src.llm.http_client_pool import get_chat_openai
                llm = get_chat_openai(temperature=0, max_tokens=10, timeout=5, max_retries=1)
            
            prompt = (
                f"判断用户是否拒绝继续。只输出一个字：Y（拒绝）或 N（不拒绝）。\n"
                f"追问、好奇、闲聊、答非所问 都算 N。\n"
                f"只有明确表示不想做、不愿意、要休息才算 Y。\n\n"
                f"用户说：「{user_input}」\n输出："
            )
            resp = llm.invoke([{"role": "user", "content": prompt}])
            answer = resp.content.strip().upper() if hasattr(resp, 'content') else str(resp).strip().upper()
            elapsed = time.time() - t0
            
            is_reject = answer.startswith("Y")
            self._log_summary_card(
                "Consent Gate",
                {"source": "llm", "decision": "declined" if is_reject else "granted", "elapsed": f"{elapsed:.2f}s"},
            )
            return not is_reject
        except Exception as e:
            self._log_summary_card("Consent Gate", {"source": "llm", "decision": "default_granted", "error": str(e)})
            return True
    def _check_mmse_complete(self, session_id: str) -> bool:
        """
        检查 MMSE 是否已完成所有必要维度
        返回 True 表示可以结束对话
        """
        # 核心评估任务（不含缓冲任务）
        core_tasks = [t for t in self.catalog.required_tasks if t not in self.catalog.buffer_tasks]
        completed_core = self.state._task_done & set(core_tasks)
        
        coverage = len(completed_core) / len(core_tasks) if core_tasks else 1.0

        self._log_summary_card(
            "MMSE Coverage",
            {
                "completed": f"{len(completed_core)}/{len(core_tasks)}",
                "coverage": f"{coverage:.0%}",
                "remaining": len(set(core_tasks) - completed_core),
            },
        )
        self._log_verbose(f"已完成: {completed_core}")
        self._log_verbose(f"未完成: {set(core_tasks) - completed_core}")
        
        # 至少完成 80% 的核心任务才算完成
        return coverage >= 1
    def _generate_soft_continuation(
        self, patient_profile: Dict, user_answer: str, chat_history: List
    ) -> str:
        """
        当用户想结束但 MMSE 未完成时，用 LLM 生成自然的过渡话语
        """
        patient_name = patient_profile.get('name', '')
        
        try:
            if self.use_local:
                from src.llm.model_pool import get_pooled_llm
                llm = get_pooled_llm(pool_key='7b_complex')
            else:
                from src.llm.http_client_pool import get_chat_openai
                llm = get_chat_openai(temperature=0.7, max_tokens=80, timeout=10, max_retries=1)

            # 获取最近几轮对话作为上下文
            recent_history = chat_history[-6:] if chat_history else []
            history_text = "\n".join([
                f"{'我' if h.get('role') == 'assistant' else '老人'}: {h.get('content', '')[:50]}"
                for h in recent_history
            ])
            
            prompt = f"""你是{patient_name or '老人'}的晚辈，正在陪他/她聊天。
老人刚才说想结束对话：「{user_answer}」

但你想继续聊一会儿。请生成一句自然的回复，要求：
1. 先温柔地回应老人（"好嘞"/"行"/"没事"等）
2. 然后自然地抛出一个新话题继续聊（不要太刻意）
3. 话题可以是：天气、吃饭、身体、家人、兴趣爱好等
4. 口语化、接地气，不要太长（20-40字）
5. 不要问"您还有什么想聊的吗"这种刻意的话

最近对话：
{history_text}

直接输出回复，不要加引号："""
            
            response = llm.invoke([{"role": "user", "content": prompt}])
            result = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            result = result.strip('"').strip("'").strip('「').strip('」')
            
            if result and 10 < len(result) < 80:
                self._log_verbose(f"LLM 生成过渡语: {result}")
                return result
        except Exception as e:
            self._log_summary_card("Soft Continuation", {"status": "fallback_template", "error": str(e)})
        
        # LLM 失败时的备选模板
        import random
        templates = [
            f"{patient_name}，好嘞，那您歇会儿。诶，您今天身体咋样？",
            f"行，{patient_name}您休息。对了，最近睡眠怎么样？",
            f"好，{patient_name}您先歇着。说起来，您家那边天气怎么样？",
            f"没事{patient_name}，您累了就歇。您孩子最近来看您了吗？",
        ]
        return random.choice(templates)
