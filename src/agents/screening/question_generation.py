from __future__ import annotations

from collections.abc import Callable
import json
import time
from typing import Any, Dict, List, Optional

from .catalog import ScreeningTaskCatalog
from .state import ScreeningSessionState
from .task_planning import ScreeningTaskPlanning
from .tool_gateway import ScreeningToolGateway


class ScreeningQuestionGeneration:
    """Generate task instructions and patient-facing screening questions."""

    def __init__(
        self,
        *,
        state: ScreeningSessionState,
        catalog: ScreeningTaskCatalog,
        tools: ScreeningToolGateway,
        task_planner: ScreeningTaskPlanning,
        log_summary: Callable[[str, Dict[str, Any]], None],
        log_verbose: Callable[[str], None],
    ) -> None:
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

    def _build_special_task_fixed_target_question(
        self,
        task_id: Optional[str],
        standard_result: Optional[Dict[str, Any]] = None,
        is_followup_naming: bool = False,
    ) -> Optional[str]:
        if not task_id:
            return None

        if task_id in {"language_naming_watch", "language_naming_pencil"}:
            return "我再给您看一张图片，请您说说这是什么？" if is_followup_naming else "我给您看一张图片，请您说说这是什么？"

        if task_id == "registration_3words":
            words = (standard_result or {}).get('memory_words') or self.state.session_data.get('memory_words') or []
            words_text = '、'.join(str(word).strip() for word in words if str(word).strip())
            if words_text:
                return f"我说三个词，您听好了：{words_text}。请您复述一遍。"
            return "我说三个词，您听好了。请您复述一遍。"

        if task_id == "recall_3words":
            return "刚才我说的那三个词，您现在还能想起来吗？请您试着回忆一下。"

        if task_id == "attention_calc_life_math":
            calc_cfg = (standard_result or {}).get('calculation_config') or self.state.session_data.get('calculation_config') or {}
            start = calc_cfg.get('start')
            step = calc_cfg.get('step', 7)
            if start is None:
                start = self.state._calculation_current_value
            if start is None:
                start = 100
            return f"要是从{start}开始，每次减{step}，请您连续减5次，把每次剩下的数都说出来。"

        if task_id == "attention_reverse_phrase":
            reverse_data = (standard_result or {}).get('data') or {}
            prompt_chars = reverse_data.get('prompt_chars') or ['祝', '出', '入', '平', '安']
            prompt_text = '、'.join(str(ch).strip() for ch in prompt_chars if str(ch).strip())
            if prompt_text:
                return f"我读几个字，您倒着说一遍：{prompt_text}。"
            return "我读几个字，您倒着说一遍。"

        if task_id == "language_repetition_sentence":
            repetition_data = (standard_result or {}).get('data') or {}
            sentence = repetition_data.get('standard_sentence') or (standard_result or {}).get('expected_answer') or '非如果，还有，或但是'
            return f"我说一句话，您跟着重复一遍就行：\"{sentence}\"。"

        return None
    def _call_fixed_target_question_generation(
        self,
        dimension_name: str,
        patient_profile: Dict,
        conversation_history: List,
        task_instruction: Optional[str],
        persona_hooks: Optional[List[str]],
        must_include: Optional[List[str]],
        patient_emotion: str,
        task_id: Optional[str],
        fixed_target_question: Optional[str],
    ) -> str:
        target = (fixed_target_question or '').strip()
        if not target:
            return "请继续"

        generated = self.tool_gateway._call_question_generation(
            dimension_name,
            "",
            patient_profile,
            conversation_history,
            False,
            is_dimension_switch=True,
            needs_encouragement=False,
            resistance_info=None,
            task_instruction=task_instruction,
            persona_hooks=persona_hooks,
            must_include=must_include,
            patient_emotion=patient_emotion,
            task_id=task_id,
            explicit_target_question=target,
        )
        if generated and generated.strip() and generated.strip() != "请继续":
            return generated
        return target
    def _get_task_instruction(self, task_id: str) -> str:
        """获取任务的内部指令（给LLM看，不给用户）"""
        instructions = {
            "persona_collect_1": "自然地问问对方平时喜欢做什么、有什么爱好，收集个人信息",
            "persona_collect_2": "继续闲聊，了解对方的生活习惯、家庭情况等",
            "buffer_chat": "纯闲聊，聊聊天气、新闻、生活琐事，不做任何评估",
            "orientation_time_year": "自然地聊到今年是哪一年，比如'对了，今年是哪一年来着？'。⚠️只问年份",
            "orientation_time_season": "自然地聊到现在什么季节，比如'这会儿是什么季节呢？'。⚠️只问季节",
            "orientation_time_month_date": "聊聊今天几月几号。⚠️只问月份和日期，不要问年份、星期、季节",
            "orientation_time_weekday": "自然地聊到今天星期几，比如'今天周几来着？'。⚠️只问星期几",
            "orientation_place_province_city": "围绕系统记录的手填地址，自然地问当前所在医院在什么省、什么市。⚠️锚定手填地址和当前医院，只问省和市，不要问老家、住址或平时住哪",
            "orientation_place_district": "围绕系统记录的手填地址，自然地问当前所在医院在什么区/县。⚠️锚定手填地址和当前医院，只问区或县，不要问老家、住址或平时住哪",
            "orientation_place_location_floor": "围绕系统记录的手填地址，自然地问当前所在医院叫什么、在几楼。比如'您现在这个医院叫什么，在几楼呀？'。⚠️锚定手填地址和当前医院，只问医院/地点和楼层，不要问老家、住址或常去哪里",
            "registration_3words": "告诉对方三个词，请对方帮忙记一下，稍后会问",
            "attention_calc_life_math": "一次性提出 100 连续减 7：请对方连续减5次，并把5个结果都说出来。不要分成5轮追问",
            "attention_reverse_phrase": "告诉对方你会读五个字，请对方按相反顺序倒着说出来。固定字组用‘祝、出、入、平、安’，看是否能倒着说成‘安、平、入、出、祝’",
            "language_naming_watch": "展示手表图片，问这是什么",
            "language_naming_pencil": "展示铅笔图片，问这是什么",
            "language_repetition_sentence": "请对方跟着准确复述 MMSE 表上的固定句子，不要替换成绕口令或别的短句",
            "language_reading_close_eyes": "展示阅读指令卡片，看对方是否照着做，不要求对方把字念出来；重点是看完后是否执行闭眼动作",
            "language_3step_action": "请对方执行三步动作指令：举起右手（若右手不便可允许改用左手）、握成拳头、把手放到胸前。不要改成拿纸/对折/放地上的任务，也不要改成口头回答题。",
            "language_writing_sentence": "请对方口头说一个完整的句子，要求有主语和谓语、有实际意思。不要打开画板，不要让对方手写",
            "recall_3words": "问问刚才说的那三个词还记得吗",
            "copy_pentagons": "展示两个相交的五边形图片，让对方在画板上照着画",
        }
        base = instructions.get(task_id, "自然地继续对话")

        # 动态追加已完成任务的禁止提示，防止 LLM 重复提问
        done_hints = []
        done_map = {
            "orientation_time_year": "年份",
            "orientation_time_season": "季节",
            "orientation_time_month_date": "月份/日期",
            "orientation_time_weekday": "星期几",
            "orientation_place_province_city": "省/市",
            "orientation_place_district": "区/县",
            "orientation_place_location_floor": "地点/楼层",
        }
        for done_task in self.state._task_done:
            label = done_map.get(done_task)
            if label and done_task != task_id:
                done_hints.append(label)
        if done_hints:
            base += f"\n⛔已评估过的内容（绝对不要再问）：{', '.join(done_hints)}"

        # 🔥 v1.4: 维度切换时，告诉LLM上一个任务是什么，避免误解用户最后一句话
        last_tid = self.state._last_task_id
        if last_tid and last_tid != task_id:
            last_dim = self.catalog.task_config.get(last_tid, {}).get('dimension_id')
            cur_dim = self.catalog.task_config.get(task_id, {}).get('dimension_id')
            if last_dim and cur_dim and last_dim != cur_dim:
                prev_task_labels = {
                    "attention_calc_life_math": "连续减法口算（100减7）",
                    "attention_reverse_phrase": "倒着说词组",
                    "orientation_time_year": "问年份",
                    "orientation_time_season": "问季节",
                    "orientation_time_month_date": "问月份/日期",
                    "orientation_time_weekday": "问星期几",
                    "orientation_place_province_city": "问省/市",
                    "orientation_place_district": "问区/县",
                    "orientation_place_location_floor": "问地点/楼层",
                    "registration_3words": "记忆三个词",
                    "recall_3words": "回忆三个词",
                }
                prev_label = prev_task_labels.get(last_tid)
                if prev_label:
                    base += f"\n⚠️上一轮刚做完「{prev_label}」，对方最后一句话是那个任务的回答，不要误解为其他含义（比如不要把数字当年龄）。简短夸一下对方算得好/记得好，然后自然过渡到新话题。"

        return base
    def _generate_buffer_question(self, task_id: str, patient_profile: Dict, chat_history: List) -> str:
        """生成缓冲任务的闲聊问题 - 使用 LLM 生成自然多样的开场白"""
        
        # 根据任务类型给 LLM 不同的指引
        task_hints = {
            "persona_collect_1": "了解对方的兴趣爱好（喜欢做什么、看什么、玩什么）。⚠️注意：如果对方说没爱好，不要强行追问，可以问问年轻时喜欢干啥，或者直接聊别的",
            "persona_collect_2": "了解对方的生活习惯（作息、饮食、日常活动）。⚠️注意：如果对方否定（如不吃早饭、不起床），不要追问吃了啥，而是问原因或通过'那午饭呢'等方式自然切换话题",
            "buffer_chat": "顺着对方刚才的话题自然聊，不要突然切到无关内容"
        }
        default_hint = task_hints.get(task_id, "随便聊聊")
        
        # 🔥 提取最近对话（结构化列表而非字符串，让 LLM 有完整上下文）
        recent_history = None
        if chat_history and len(chat_history) > 0:
            recent_history = chat_history[-6:]  # 最近3轮（6条消息）
        
        # 🔥 话题优先：如果 LLM 选了话题，用话题作为主要指引
        bridge_hint = self.state._last_bridge_hint
        bridge_topic = self.state._last_bridge_topic
        bridge_hint_for_prompt = bridge_hint
        structured_eval_markers = ('星期', '周几', '几月', '几号', '日期', '季节', '哪一年', '年份', '省', '市', '区', '县', '地方', '楼')
        if task_id == "buffer_chat":
            topic_text = (bridge_topic or bridge_hint or '').strip()
            if topic_text and any(marker in topic_text for marker in structured_eval_markers):
                bridge_hint = None
                bridge_hint_for_prompt = None
                bridge_topic = None

        if bridge_hint:
            # 从 bridge_hint 提取目标话题 (格式: "A→B")
            to_topic = bridge_topic or (bridge_hint.split("→")[-1].strip() if "→" in bridge_hint else bridge_hint)
            # 用话题作为主指令，buffer_chat 不再拼接固定参考文案，避免“近况+聊天气”冲突
            if task_id == "buffer_chat":
                hint = (
                    f"围绕「{to_topic}」这个话题自然聊天。"
                    f"顺着对方刚才的话往下聊，不要切到与「{to_topic}」无关的话题。"
                )
            else:
                hint = f"围绕「{to_topic}」这个话题自然聊天。可以参考：{default_hint}"
            # 🔥 注入防重复指令
            last_user_content = chat_history[-1].get('content', '') if chat_history else ''
            if last_user_content:
                hint += f"\n【注意】对方刚才说了「{last_user_content[:30]}...」，请顺着这话往下聊，不要重复问对方已经说过的信息。"
            
            self._log_summary_card("Buffer Question", {"topic": to_topic, "task": task_id, "mode": "topic_priority"})
        else:
            hint = default_hint
            if task_id == "buffer_chat":
                hint += "\n【注意】只顺着对方刚才的话自然聊，不要重复刚做完的测评题点（如星期几、日期、季节、年份、地点）。"

        if task_id in self.catalog.buffer_tasks:
            last_ai_q = self.task_planner._extract_last_assistant_question(chat_history)
            if last_ai_q:
                hint += f"\n【上一轮助手问题】{last_ai_q}"
            recent_asked = [q for q in self.state._asked_questions[-6:] if q]
            if recent_asked:
                hint += f"\n【最近已问过】{'；'.join(recent_asked)}"
            hint += (
                "\n【强约束】不要复问已经问过的信息。"
                "像坐着舒不舒服、累不累、撑不撑得住算同一类；"
                "吃没吃早饭/午饭/晚饭算同一类；天气、季节、外头亮不亮算同一类。"
                "这些都不能只换个说法再问，必须换一个新的信息点。"
            )
        
        # 🔥 流式回调注入
        stream_cb = self.state._stream_sentence_cb
        if stream_cb:
            self.tool_gateway.question_tool._on_sentence_cb = stream_cb

        # 调用 QuestionGenerationTool（传入结构化历史）
        try:
            result_json = self.tool_gateway.question_tool._run(
                dimension_name="闲聊",
                dimension_description=hint,
                patient_name=patient_profile.get('name'),
                patient_gender=patient_profile.get('gender'),
                patient_age=patient_profile.get('age'),
                conversation_history=recent_history,
                task_instruction=hint,
                task_id=task_id,
                avoid_questions=self.state._asked_questions,
                bridge_hint=bridge_hint_for_prompt,
            )
        finally:
            self.tool_gateway.question_tool._on_sentence_cb = None
        
        try:
            result = json.loads(result_json)
            if result.get('success') and result.get('question'):
                question = result['question']
                return question
        except:
            pass
        
        # 兜底：如果 LLM 失败，用简单的开场
        patient_name = patient_profile.get('name', '')
        greeting = f"{patient_name}，" if patient_name else ""
        return f"{greeting}您最近怎么样？"
    def _extract_persona_hooks(self, patient_profile: Dict, chat_history: List) -> List[str]:
        hooks = []
        for key in [
            'hobby', 'hobbies', 'interest', 'interests', 'occupation', 'job', 'hometown',
            'city', 'district', 'nickname'
        ]:
            val = patient_profile.get(key)
            if isinstance(val, str) and val.strip():
                hooks.append(val.strip())
            elif isinstance(val, list):
                hooks.extend([str(x).strip() for x in val if str(x).strip()])

        if chat_history:
            last_user = None
            for msg in reversed(chat_history):
                if msg.get('role') == 'user':
                    last_user = (msg.get('content') or '').strip()
                    break
            if last_user and len(last_user) <= 20:
                hooks.append(last_user)

        # 去重
        dedup = []
        seen = set()
        for h in hooks:
            if h not in seen:
                dedup.append(h)
                seen.add(h)
        return dedup[:5]
    def _filter_persona_hooks_for_task(self, task_id: str, hooks: Optional[List[str]]) -> List[str]:
        hooks = list(hooks or [])
        if not hooks:
            return []

        if task_id.startswith("orientation_"):
            return []

        return hooks[:5]
    def _get_must_include_for_task(self, task_id: str) -> Optional[List[str]]:
        if task_id == 'orientation_time_year':
            return ['哪一年', '年']
        if task_id == 'orientation_time_season':
            return ['什么季节']
        if task_id == 'orientation_time_month_date':
            return ['几月', '几号']
        if task_id == 'orientation_time_weekday':
            return ['星期几']
        if task_id == 'orientation_place_province_city':
            return ['省', '市']
        if task_id == 'orientation_place_district':
            return ['区', '县']
        if task_id == 'orientation_place_location_floor':
            return ['地方', '楼']
        if task_id == 'language_reading_close_eyes':
            return ['照着做']
        if task_id == 'language_3step_action':
            return ['右手', '握拳', '胸前']
        if task_id == 'language_writing_sentence':
            return ['说', '句子']
        if task_id == 'copy_pentagons':
            return ['画', '照着']
        return None
    def _generate_assessment_question(
        self, dimension_name: str, patient_profile: Dict, 
        chat_history: List, start_time: float,
        task_id: Optional[str] = None  # 🔥 新增：任务ID
    ) -> Dict[str, Any]:
        """
        生成评估问题（从闲聊回到评估）
        重新走正常的问题生成流程
        """
        # 生成下一个问题（正常流程）
        self._log_summary_card("Assessment Resume", {"task": task_id, "dimension": dimension_name})
        
        # 🔥 获取任务指令（如果有任务ID）
        task_instruction = None
        if task_id:
            task_instruction = self._get_task_instruction(task_id)
            self._log_verbose(f"任务指令: {task_instruction[:50] if task_instruction else 'None'}...")
        
        # 生成检索查询
        query_result = self.tool_gateway._call_query_generation(
            dimension_name,
            chat_history,
        )
        
        # 检索知识
        retrieval_result = self.tool_gateway._call_knowledge_retrieval(
            query_result
        )
        
        # 生成问题（从闲聊回到评估，标记为维度切换以便生成自然过渡）
        next_question = self.tool_gateway._call_question_generation(
            dimension_name,
            retrieval_result.get('knowledge_context', ''),
            patient_profile,
            chat_history,
            is_followup=False,
            is_dimension_switch=True,  # 🔥 标记为维度切换，触发过渡语生成
            needs_encouragement=False,
            resistance_info=None,
            task_instruction=task_instruction,  # 🔥 传入任务指令
            task_id=task_id,
        )
        
        # 问题生成已经包含对用户回答的回应，不需要额外过渡语
        
        return {
            'output': next_question,
            'response': next_question,
            'is_comfort_mode': False,
            'returned_to_assessment': True,
            'dimension': dimension_name,
            'total_time': time.time() - start_time
        }
