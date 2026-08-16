"""
特殊维度问题工具 - 用 LLM 生成自然问题 + 结构化输出

处理 MMSE 中需要特殊处理的维度：
- registration (即时记忆): LLM 生成问题，提取记忆词
- attention_calculation (注意力计算): LLM 生成问题，提取计算参数
- recall (延迟回忆): 基于之前的记忆词生成问题
- copy (临摹): 生成引导语 + 展示图片
"""

import json
import os
import re
import random
from typing import Optional, Type, Dict, Any, List
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI


# ==================== 维度配置 ====================
SPECIAL_DIMENSIONS: Dict[str, Dict[str, Any]] = {
    # 原有标准化任务
    'registration': {
        'trigger': 'on_switch',
        'purpose': '测试即时记忆能力',
        'log_tag': '📝 即时记忆',
        'default_words': ['苹果', '桌子', '硬币'],
        'word_pools': [
            ['苹果', '桌子', '硬币'],
            ['香蕉', '椅子', '钥匙'],
            ['西瓜', '电话', '手表'],
            ['橘子', '窗户', '雨伞'],
        ],
    },
    'attention_calculation': {
        'trigger': 'always',
        'purpose': '测试注意力和计算能力',
        'log_tag': '🔢 注意力计算',
        'default_config': {'start': 100, 'step': 7},
    },
    'attention_reverse_phrase': {
        'trigger': 'on_switch',
        'purpose': '测试注意力和倒序复述能力',
        'log_tag': '🔁 倒着说词组',
        'prompt_chars': ['祝', '出', '入', '平', '安'],
    },
    'recall': {
        'trigger': 'on_switch',
        'purpose': '测试延迟记忆能力',
        'log_tag': '🧠 延迟回忆',
    },
    'copy': {
        'trigger': 'always',
        'purpose': '测试视觉空间构造能力',
        'log_tag': '📋 临摹',
        'requires_image': True,
        'image_id': 'pentagons',
        'image_title': '请看这两个图形，试着在纸上把它们画下来',
    },
    # 🔥 语言任务（复述句子）
    'language_repetition': {
        'trigger': 'on_switch',
        'purpose': '测试语言复述能力',
        'log_tag': '🗣️ 复述句子',
        'standard_sentence': '非如果，还有，或但是',  # 标准复述句子
        'instruction': '请跟我说一遍这句话："非如果，还有，或但是"',
    },
}


class StandardQuestionToolArgs(BaseModel):
    """特殊维度问题工具参数"""
    dimension_id: str = Field(
        ..., 
        description="维度ID: registration, attention_calculation, recall, copy, orientation, language"
    )
    is_dimension_switch: bool = Field(
        default=False,
        description="是否刚切换到此维度"
    )
    memory_words: Optional[List[str]] = Field(
        default=None,
        description="之前记忆的词（用于 recall 维度）"
    )
    patient_name: Optional[str] = Field(
        default=None,
        description="患者姓名（用于个性化）"
    )
    calculation_current_value: Optional[int] = Field(
        default=None,
        description="连续减法的当前值（用于 attention_calculation）"
    )
    calculation_step: Optional[int] = Field(
        default=7,
        description="连续减法的步长（默认7）"
    )
    last_user_message: Optional[str] = Field(
        default=None,
        description="用户上一轮说的话，用于生成自然的回应过渡"
    )


class StandardQuestionTool(BaseTool):
    """
    特殊维度问题生成工具
    
    用 LLM 生成自然的问题，同时返回结构化的关键数据。
    确保问题自然流畅，同时关键信息可追踪。
    """
    
    name: str = "StandardQuestionTool"
    description: str = """
    为 MMSE 特殊维度生成自然的问题。
    
    与普通问题生成不同，这个工具会：
    1. 生成自然流畅的问题
    2. 返回结构化的关键数据（如记忆词、计算参数）
    3. 确保 recall 时使用的词与 registration 一致
    """
    args_schema: Type[BaseModel] = StandardQuestionToolArgs
    
    _llm: Any = PrivateAttr()
    _use_local: bool = PrivateAttr()
    
    def __init__(self, use_local: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._use_local = use_local
        
        if use_local:
            # 使用本地模型池
            from src.llm.model_pool import get_pooled_llm
            self._llm = get_pooled_llm(pool_key='7b_complex')  # 用 7B 模型生成自然问题
            print("[StandardQuestionTool] 🏠 使用本地模型 (7b_complex)")
        else:
            from src.llm.http_client_pool import get_chat_openai

            fallback_model = None
            if not os.getenv("DASHSCOPE_API_KEY"):
                fallback_model = (
                    "doubao-seed-2-0-mini-260215"
                    if os.getenv("ARK_API_KEY")
                    else "Qwen/Qwen2.5-7B-Instruct"
                )
            self._llm = get_chat_openai(
                model=fallback_model,
                temperature=0.7,
                max_tokens=200,
                timeout=10,
                max_retries=1,
            )
            print("[StandardQuestionTool] 使用统一 LLM 路由")
    
    def _extract_response(self, response) -> str:
        """从 LLM 响应中提取文本（兼容本地模型和 API）"""
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)

    def _run(
        self,
        dimension_id: str,
        is_dimension_switch: bool = False,
        memory_words: Optional[List[str]] = None,
        patient_name: Optional[str] = None,
        calculation_current_value: Optional[int] = None,
        calculation_step: Optional[int] = 7,
        last_user_message: Optional[str] = None,
    ) -> str:
        """
        生成特殊维度的问题
        
        Returns:
            JSON 格式，包含 question 和结构化数据
        """
        
        # 检查是否是特殊维度
        if dimension_id not in SPECIAL_DIMENSIONS:
            return json.dumps({
                "has_standard_question": False,
                "dimension_id": dimension_id,
                "message": "此维度使用普通问题生成"
            }, ensure_ascii=False)
        
        config = SPECIAL_DIMENSIONS[dimension_id]
        trigger = config.get('trigger', 'on_switch')
        
        # 检查触发条件
        should_trigger = (trigger == 'always') or (trigger == 'on_switch' and is_dimension_switch)
        
        if not should_trigger:
            return json.dumps({
                "has_standard_question": False,
                "dimension_id": dimension_id,
                "message": "不满足触发条件（非首次进入维度）"
            }, ensure_ascii=False)
        
        print(f"[StandardQuestion] {config['log_tag']} - 生成问题...")
        
        # 根据维度类型生成问题
        if dimension_id == 'registration':
            return self._generate_registration_question(config, patient_name, last_user_message)
        elif dimension_id == 'attention_calculation':
            return self._generate_calculation_question(config, patient_name, calculation_current_value, calculation_step, last_user_message)
        elif dimension_id == 'attention_reverse_phrase':
            return self._generate_reverse_phrase_question(config, patient_name, last_user_message)
        elif dimension_id == 'recall':
            return self._generate_recall_question(config, memory_words, patient_name, last_user_message)
        elif dimension_id == 'copy':
            return self._generate_copy_question(config, patient_name, last_user_message)
        elif dimension_id == 'language_repetition':
            # 🔥 语言复述任务：返回固定句子
            return self._generate_repetition_question(config, patient_name, last_user_message)
        elif dimension_id in ['orientation', 'language']:
            # 新的灵活任务：交给QuestionGenerationTool处理
            return json.dumps({
                "has_standard_question": False,
                "dimension_id": dimension_id,
                "message": f"灵活任务{dimension_id}应由QuestionGenerationTool处理"
            }, ensure_ascii=False)
        
        return json.dumps({"has_standard_question": False}, ensure_ascii=False)
    
    def _generate_registration_question(self, config: dict, patient_name: Optional[str], last_user_message: Optional[str] = None) -> str:
        """生成即时记忆问题 - LLM 自主选择三个词"""
        
        context_info = f"患者叫{patient_name}。" if patient_name else ""

        
        # 构建对话上下文
        ack_instruction = ""
        if last_user_message and last_user_message.strip():
            ack_instruction = (
                f'\n5. 对方刚说了「{last_user_message.strip()}」。'
                f'请像朋友一样，先自然接一小句他话里的具体信息（不要照搬原话，也不要用"对了""嗯呐""话说""说到这个"这类万能过渡词），再顺口引出记三个词。'
                f'承接的话题要像诊室里陪老人轻松聊天，只围绕当下容易回答的事（坐着舒不舒服、眼前天气、刚才休息得怎样、中午吃没吃、现在这个房间/楼层）。不要聊候诊、排队、怎么来的，也不要默认住院、病房、食堂、陪护、住几天。'
            )
        
        prompt = f"""你是在诊室里陪老人轻松聊天的医生助理。{context_info}
现在需要在不强调测评的情况下，请患者顺手记住三个词并复述。

要求：
1. **自主选择三个词**，必须满足：
   - 互不相关（不同类别，如：水果、家具、交通工具）
   - 非常常见、具体、易懂（避免抽象词）
   - 必须是中文双字词（如：苹果、桌子、硬币）
2. 用温和亲切的语气引导，如果是长辈可以用"您"
3. 清晰地说出这三个词，并让老人复述一遍
4. **仅输出JSON格式**，不要包含Markdown标记或其他文字{ack_instruction}

好的词例：
- 苹果、桌子、硬币
- 香蕉、椅子、钥匙
- 西瓜、电话、手表

请严格按此JSON格式回复：
{{"question": "生成的问句", "words": ["词1", "词2", "词3"]}}"""

        # 默认回退值
        default_words = config.get('default_words', ['苹果', '桌子', '硬币'])
        words_str = '、'.join(default_words)
        fallback_question = f"我说三个词，请您先认真听一下，然后跟着我复述一遍：{words_str}。请您说一遍。"
        
        try:
            response = self._llm.invoke([{"role": "user", "content": prompt}])
            response_text = self._extract_response(response).strip()
            
            # 清理 Markdown 代码块
            if "```" in response_text:
                import re
                match = re.search(r"```(?:json)?(.*?)```", response_text, re.DOTALL)
                if match:
                    response_text = match.group(1).strip()
            
            # 尝试解析 JSON
            words = None
            question = None
            
            import re
            # 寻找最外层的 {} 
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    question = parsed.get('question', '').strip()
                    words = parsed.get('words', [])
                except json.JSONDecodeError:
                    print(f"[StandardQuestion] ⚠️ JSON解析错误: response_chars={len(response_text)}")
            
            # 验证数据的有效性
            if not words or not isinstance(words, list) or len(words) != 3:
                print(f"[StandardQuestion] ⚠️ 生成的词无效: word_count={len(words) if isinstance(words, list) else 0}，使用备选词池")
                words = random.choice(config.get('word_pools', [default_words]))
                words_str = '、'.join(words)
                # 重新构建问题以确保包含这些词
                if patient_name:
                    question = f"{patient_name}，我说三个词，您听好了：{words_str}。请您复述一遍？"
                else:
                    question = f"我说三个词，您听好了：{words_str}。请您复述一遍？"
            
            # 确保词语在问题中
            if question and not all(w in question for w in words):
                words_str = '、'.join(words)
                if patient_name:
                    question = f"{patient_name}，我说三个词，您听好了：{words_str}。请您复述一遍？"
                else:
                    question = f"我说三个词，您听好了：{words_str}。请您复述一遍？"
            
            print(f"[StandardQuestion] ✅ 生成问题: text_chars={len(question or '')}")
            print(f"[StandardQuestion] 📝 LLM选择的词: word_count={len(words)}")
            
            return json.dumps({
                "has_standard_question": True,
                "dimension_id": "registration",
                "question": question,
                "memory_words": words,  # ⭐ 关键：返回记忆词，供 recall 使用
                "log_tag": config['log_tag'],
                "requires_image": False,
            }, ensure_ascii=False)
            
        except Exception as e:
            # ❗ 调试模式：不用兆底，直接报错
            print(f"[StandardQuestion] ❌ LLM 生成失败: {type(e).__name__}")
            raise e  # 让错误暴露出来
    
    def _generate_calculation_question(
        self, 
        config: dict, 
        patient_name: Optional[str],
        current_value: Optional[int] = None,
        step: Optional[int] = 7,
        last_user_message: Optional[str] = None
    ) -> str:
        """生成注意力计算问题：一次性要求 100 连续减 7 共 5 次。"""

        start = 100
        step = step or 7
        expected_sequence = [start - step * idx for idx in range(1, 6)]
        expected_answer = "、".join(str(num) for num in expected_sequence)
        
        context_info = f"患者叫{patient_name}。" if patient_name else ""
        
        ack_part = ""
        if last_user_message and last_user_message.strip():
            ack_part = (
                f'\n- 对方刚说了「{last_user_message.strip()}」。'
                f'请像朋友一样，先自然接一小句他话里的具体信息（不用"对了""嗯呐""话说"这类万能过渡词），再引出连续减法。'
                f'承接只能用诊室里当下容易回答的内容（坐着舒不舒服、中午吃没吃、昨晚睡得怎样、眼前天气、现在这个房间/楼层），不要聊候诊、排队、怎么来的，也不要带出住院、病房、食堂、陪护、住几天这些没说过的设定。'
            )
        
        prompt = f"""你是在诊室里陪老人轻松聊天的医生助理。{context_info}
请自然地一次性提出连续减法任务：从{start}开始，每次减{step}，请对方连续减5次，并把每次的结果都说出来。
不要拆成一问一答，不要只问第一步，不要说"测评"、"测试一下"。

要求：一句话，口语化，必须包含数字{start}、{step}和"连续/一直/接着"这类表达，并明确要说出5次结果。仅输出JSON。{ack_part}
示例：
- "那我请您心里算一下，从{start}开始，每次减{step}，连续减5次，把每次剩下的数都说出来，好吗？"
- "咱们就按{start}往下算，每次减{step}，您接着说5个结果就行。"

{{"question": "生成的问句"}}"""

        try:
            response = self._llm.invoke([{"role": "user", "content": prompt}])
            response_text = self._extract_response(response).strip()
            
            # 清理 Markdown
            if "```" in response_text:
                import re
                match = re.search(r"```(?:json)?(.*?)```", response_text, re.DOTALL)
                if match:
                    response_text = match.group(1).strip()
            
            question = None
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    question = parsed.get('question', '').strip()
                except:
                    pass
            
            # Fallback if JSON fails or question is empty
            if not question:
                question = response_text.strip().strip('"').strip("'")
                if len(question) > 50 or "{" in question:
                    question = None

            # 验证关键信息
            if question:
                if str(start) not in question or str(step) not in question or not any(token in question for token in ("5", "五")):
                    question = f"那我请您心里算一下，从{start}开始，每次减{step}，连续减5次，把每次剩下的数都说出来，好吗？"
            else:
                question = f"那我请您心里算一下，从{start}开始，每次减{step}，连续减5次，把每次剩下的数都说出来，好吗？"
            
            print(f"[StandardQuestion] ✅ 生成问题: text_chars={len(question or '')}, expected_chars={len(str(expected_answer))}")
            
            return json.dumps({
                "has_standard_question": True,
                "dimension_id": "attention_calculation",
                "question": question,
                "calculation_config": {
                    "start": start,
                    "step": step,
                    "expected_answer": expected_answer,
                    "expected_sequence": expected_sequence,
                },
                "log_tag": config['log_tag'],
                "requires_image": False,
            }, ensure_ascii=False)
            
        except Exception as e:
            print(f"[StandardQuestion] ❌ LLM 生成失败: {type(e).__name__}")
            raise e

    def _generate_reverse_phrase_question(
        self,
        config: dict,
        patient_name: Optional[str],
        last_user_message: Optional[str] = None,
    ) -> str:
        """生成注意力倒序复述问题 - 固定词组"""

        prompt_chars = [str(ch).strip() for ch in (config.get('prompt_chars') or ['祝', '出', '入', '平', '安']) if str(ch).strip()]
        if not prompt_chars:
            prompt_chars = ['祝', '出', '入', '平', '安']
        expected_chars = list(reversed(prompt_chars))
        prompt_text = '、'.join(prompt_chars)
        expected_answer = ''.join(expected_chars)

        if patient_name:
            question = f'{patient_name}，我读几个字给您听，请您倒着说出来：{prompt_text}。'
        else:
            question = f'我读几个字给您听，请您倒着说出来：{prompt_text}。'

        return json.dumps({
            "has_standard_question": True,
            "dimension_id": "attention_reverse_phrase",
            "question": question,
            "expected_answer": expected_answer,
            "data": {
                "prompt_chars": prompt_chars,
                "prompt_text": prompt_text,
                "expected_chars": expected_chars,
                "task_type": "reverse_phrase"
            },
            "log_tag": config['log_tag'],
            "requires_image": False,
        }, ensure_ascii=False)
    
    def _generate_recall_question(
        self, config: dict, memory_words: Optional[List[str]], patient_name: Optional[str],
        last_user_message: Optional[str] = None
    ) -> str:
        """生成延迟回忆问题"""
        
        if not memory_words:
            # 没有记忆词，用默认的
            memory_words = ['苹果', '桌子', '硬币']
            print("[StandardQuestion] ⚠️ 没有传入 memory_words，使用默认词")
        
        context_info = f"患者叫{patient_name}。" if patient_name else ""
        
        ack_instruction = ""
        if last_user_message and last_user_message.strip():
            ack_instruction = (
                f'\n5. 对方刚说了「{last_user_message.strip()}」。'
                f'请像朋友一样，先自然接一小句他话里的具体信息（不用"对了""嗯呐""话说"这类万能过渡词），再引出刚才让他记的那三样东西。'
                f'承接只能用诊室里当下容易回答的内容（坐着舒不舒服、中午吃没吃、昨晚睡得怎样、眼前天气、现在这个房间/楼层），不要聊候诊、排队、怎么来的，也不要默认住院、病房、食堂、陪护、住几天。'
            )
        
        prompt = f"""你是在诊室里陪老人轻松聊天的医生助理。{context_info}
之前让患者记住了三个词：{', '.join(memory_words)}
现在需要自然地问患者是否还记得这三个词，不要说成考试或测评。

要求：
1. 用温和的语气询问，不要给老人压力
2. **绝对不要**直接说出那三个词，而是问"刚才那三个词"或"刚才让您记的东西"
3. 给老人鼓励，让他们尝试回忆
4. **仅输出JSON格式**{ack_instruction}

示例风格：
- "刚才我说的那三个词，您现在还能想起来吗？"
- "您还记得刚才让您记的那三样东西吗？请您试着回忆一下。"

请严格按此JSON格式回复：
{{"question": "生成的问句"}}"""

        try:
            response = self._llm.invoke([{"role": "user", "content": prompt}])
            response_text = self._extract_response(response).strip()
            
            # 清理 Markdown
            if "```" in response_text:
                import re
                match = re.search(r"```(?:json)?(.*?)```", response_text, re.DOTALL)
                if match:
                    response_text = match.group(1).strip()
            
            question = None
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    question = parsed.get('question', '').strip()
                except:
                    pass
            
            # Fallback if JSON fails or question is empty
            if not question:
                question = response_text.strip().strip('"').strip("'")
                if len(question) > 50 or "{" in question:
                    question = None
            
            # 确保问题中没有直接说出答案（防止泄漏）
            if question and any(w in question for w in memory_words):
                print(f"[StandardQuestion] ⚠️ 生成的问题泄漏了答案，使用默认问题")
                if patient_name:
                    question = f"{patient_name}，刚才我说的那三个词，您现在还能想起来吗？请您试着回忆一下。"
                else:
                    question = "刚才我说的那三个词，您现在还能想起来吗？请您试着回忆一下。"
            
            # 最终兜底
            if not question:
                if patient_name:
                    question = f"{patient_name}，刚才我说的那三个词，您现在还能想起来吗？请您试着回忆一下。"
                else:
                    question = "刚才我说的那三个词，您现在还能想起来吗？请您试着回忆一下。"
            
            print(f"[StandardQuestion] ✅ 生成问题: text_chars={len(question or '')}")
            
            return json.dumps({
                "has_standard_question": True,
                "dimension_id": "recall",
                "question": question,
                "expected_words": memory_words,  # ⭐ 返回期望的答案，用于评分
                "log_tag": config['log_tag'],
                "requires_image": False,
            }, ensure_ascii=False)
            
        except Exception as e:
            # ❗ 调试模式：不用兆底，直接报错
            print(f"[StandardQuestion] ❌ LLM 生成失败: {type(e).__name__}")
            raise e
    
    def _generate_repetition_question(
        self,
        config: dict,
        patient_name: Optional[str],
        last_user_message: Optional[str] = None,
    ) -> str:
        """生成语言复述问题 - 固定句子"""
        
        sentence = config.get('standard_sentence', '非如果，还有，或但是')
        
        if patient_name:
            question = f'{patient_name}，我说一句话，您跟着重复一遍就行："{sentence}"'
        else:
            question = f'我说一句话，您跟着重复一遍就行："{sentence}"'

        return json.dumps({
            "has_standard_question": True,
            "dimension_id": "language_repetition",
            "question": question,
            "expected_answer": sentence,
            "data": {
                "standard_sentence": sentence,
                "task_type": "repetition"
            }
        }, ensure_ascii=False)
    
    def _generate_copy_question(
        self,
        config: dict,
        patient_name: Optional[str],
        last_user_message: Optional[str] = None,
    ) -> str:
        """生成临摹问题"""
        
        context_info = f"患者叫{patient_name}。" if patient_name else ""
        ack_instruction = ""
        if last_user_message and last_user_message.strip():
            ack_instruction = (
                f'\n6. 对方刚说了「{last_user_message.strip()}」。'
                f'请像朋友一样，先自然接一小句他话里的具体信息（不用"对了""嗯呐""话说"这类万能过渡词），再引导他看屏幕上的图形、在纸上画下来。'
                f'承接只能用诊室里当下容易回答的内容（坐着舒不舒服、眼前天气、昨晚睡得怎样、现在这个房间/楼层），不要聊候诊、排队、怎么来的，也不要默认住院、病房、食堂、陪护、住几天。'
            )
        
        prompt = f"""你是在诊室里陪老人轻松聊天的医生助理。{context_info}
现在需要自然地请患者看屏幕上的图形，然后在纸上画下来，不要说成测试。

要求：
1. 用轻松的语气引导
2. 必须明确说明"看屏幕上的图形"
3. 必须明确说明"在纸上画下来"
4. 一句话，不要太长
5. **仅输出JSON格式**{ack_instruction}

示例风格：
- "接下来请您看一下屏幕上的两个图形，并在纸上画下来。"
- "请您看看屏幕，试着把这两个图形画在纸上，画好后告诉我。"

请严格按此JSON格式回复：
{{"question": "生成的问句"}}"""

        try:
            response = self._llm.invoke([{"role": "user", "content": prompt}])
            response_text = self._extract_response(response).strip()
            
            # 清理 Markdown
            if "```" in response_text:
                import re
                match = re.search(r"```(?:json)?(.*?)```", response_text, re.DOTALL)
                if match:
                    response_text = match.group(1).strip()
            
            question = None
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    question = parsed.get('question', '').strip()
                except:
                    pass
            
            # Fallback
            if not question:
                question = response_text.strip().strip('"').strip("'")
                if len(question) > 50 or "{" in question:
                    question = None
            
            if not question:
                if patient_name:
                    question = f"{patient_name}，接下来请您看一下屏幕上的两个图形，并在纸上画下来。"
                else:
                    question = "接下来请您看一下屏幕上的两个图形，并在纸上画下来。"
            
            print(f"[StandardQuestion] ✅ 生成问题: text_chars={len(question or '')}")
            
            return json.dumps({
                "has_standard_question": True,
                "dimension_id": "copy",
                "question": question,
                "log_tag": config['log_tag'],
                "requires_image": True,
                "image_config": {
                    "image_id": config.get('image_id', 'pentagons'),
                    "title": config.get('image_title', '请看这两个图形'),
                },
            }, ensure_ascii=False)
            
        except Exception as e:
            # ❗ 调试模式：不用兆底，直接报错
            print(f"[StandardQuestion] ❌ LLM 生成失败: {type(e).__name__}")
            raise e
    
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("StandardQuestionTool 不支持异步调用")
