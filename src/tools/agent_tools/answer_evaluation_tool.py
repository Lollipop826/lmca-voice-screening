"""
回答评估工具 - 评估患者回答的正确性和质量

简化版：移除未使用的字段，统一代码风格
"""

from __future__ import annotations

import os
import json
import re
import time
from typing import Optional, Type, Dict, Any, ClassVar

try:
    from pypinyin import lazy_pinyin
except Exception:  # pragma: no cover - pypinyin may be absent in slim deployments
    lazy_pinyin = None

from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from src.utils.location_service import get_realtime_context
from src.llm.http_client_pool import get_siliconflow_chat_openai, get_volcengine_chat_openai
from src.utils.tool_logger import ToolLogger


class AnswerEvaluationToolArgs(BaseModel):
    """回答评估工具参数"""
    
    question: str = Field(..., description="医生提出的问题")
    answer: str = Field(default="", description="患者的回答")
    task_id: str = Field(default="", description="任务ID: orientation_time_weekday / registration_3words / attention_calc_life_math 等")
    expected_answer: Optional[str] = Field(default=None, description="期望的正确答案")
    patient_profile: Optional[Dict[str, Any]] = Field(default=None, description="患者信息")


class AnswerEvaluationTool(BaseTool):
    """
    回答评估工具
    
    评估患者回答的正确性，输出质量等级和认知表现。
    """
    
    name: str = "answer_evaluation_tool"
    description: str = "评估患者回答的质量和认知表现，返回是否正确、质量等级、认知表现等"
    args_schema: Type[BaseModel] = AnswerEvaluationToolArgs
    
    # 任务信息（包含评估提示）
    TASK_INFO: ClassVar[Dict[str, Dict[str, str]]] = {
        # 定向力任务
        "orientation_time_weekday": {
            "name": "星期几", 
            "desc": "询问今天星期几",
            "eval_hint": "判断患者说的星期几是否与当前实际日期一致。允许口语化表达如'周五'='星期五'"
        },
        "orientation_time_year_season_month_date": {
            "name": "年份/季节/月份/日期", 
            "desc": "询问今年是哪一年、什么季节、几月、几号",
            "eval_hint": "分别评估4个子项：(1)年份是否正确 (2)季节是否正确 (3)月份是否正确 (4)几号是否正确（误差1天可接受）。每个子项独立计分(对=1分,错=0分)。回答中未涉及的项算0分。季节偏好、感受描述不算答对。"
        },
        "orientation_place_full": {
            "name": "地点定向(省/市/区/医院/楼层)", 
            "desc": "询问系统记录的手填地址对应的省份、城市、区县、具体地点/医院、第几层楼",
            "eval_hint": "分别评估5个子项：(1)省份 (2)城市 (3)区/县 (4)手填地址中的具体地点/医院/科室 (5)第几层楼。每个子项独立计分(对=1分,错=0分)。回答中未涉及的项算0分。省略'省/市/区'字样可接受。"
        },
        # 记忆任务
        "registration_3words": {
            "name": "即时记忆", 
            "desc": "让患者复述三个词",
            "eval_hint": "判断患者是否准确复述了医生说的三个词。顺序可不同，但词必须完全匹配"
        },
        "recall_3words": {
            "name": "延迟回忆", 
            "desc": "回忆之前的三个词",
            "eval_hint": "判断患者回忆出了几个词。每答对一个词算部分正确，三个全对为完全正确"
        },
        # 注意力计算
        "attention_calc_life_math": {
            "name": "连续减法",
            "desc": "100连续减7，一次性说出5个结果",
            "eval_hint": (
                "【任务】患者一次性说出 100 连续减 7 共 5 次的 5 个中间结果。\n"
                "【标准答案】93、86、79、72、65（每个 1 分，共 5 分）。\n"
                "【MMSE 延续规则（重要）】\n"
                "  - 如果患者算错某一步，但接下来是从他刚才说的（错的）数继续减 7，那一步算『相对正确』，仍可记 1 分。\n"
                "    例：100→93→86→78（错，应是79）→71（=78-7，相对正确，记1分）→64（=71-7，相对正确，记1分）。\n"
                "  - 即只与上一步比对差是否为 7，判断是否给分；只有起点 100 比对的第一步固定参照 93。\n"
                "【口语/ASR 容错】\n"
                "  - '九十三'='93'，'八六'='86'；患者带语气词（嗯、啊、那个）、解释（『100减7是93』）都不扣分。\n"
                "  - 数字写法不一定完整出现，但意思清楚也算对（如『八十九』听成『八九』，结合上下文判断）。\n"
                "  - 患者只回答了部分（比如只说了 2-3 个），说出来几个就给几个分。\n"
                "  - 患者高龄、文化程度低或紧张犹豫，只要数字本身对就记分，不因犹豫扣分。\n"
                "【输出额外字段】除 is_correct/quality 外，必须输出 raw_score（0-5 的整数）和 raw_max_score=5。\n"
                "  - raw_score 即按上述规则算出的得分总数；quality 由分数推断：5=excellent, 3-4=good, 1-2=fair, 0=poor。"
            ),
        },
        "attention_reverse_phrase": {
            "name": "倒着说词组",
            "desc": "把‘祝 出 入 平 安’按相反顺序倒着说出来",
            "eval_hint": "重点判断患者是否把‘祝、出、入、平、安’正确倒着说成‘安、平、入、出、祝’。可以按字符位置计分：每个位置正确可记1分，共5分；若有轻微停顿或分开念，但顺序正确，仍应判为正确。ASR 同音字容错：只要该位置的读音相同也算正确，例如同音替换、声调差异、常见识别错字均不扣分。必须输出 raw_score（0-5 的整数）和 raw_max_score=5。"
        },
        # 语言任务
        "language_naming_watch": {
            "name": "命名-手表", 
            "desc": "说出手表的名称",
            "eval_hint": "判断是否说出'手表/表'。说'钟表/时钟'部分正确，说其他物品名称错误"
        },
        "language_naming_pencil": {
            "name": "命名-铅笔", 
            "desc": "说出铅笔的名称",
            "eval_hint": "判断是否说出'铅笔/笔'。说'钢笔/圆珠笔'部分正确，说其他物品名称错误"
        },
        "language_repetition_sentence": {
            "name": "复述句子", 
            "desc": "复述 MMSE 固定句子",
            "eval_hint": "判断是否准确复述原句。允许轻微口音差异，但字词必须基本一致"
        },
        "language_reading_close_eyes": {
            "name": "阅读指令", 
            "desc": "阅读卡片指令并执行闭眼动作",
            "eval_hint": "重点判断患者是否按卡片指令闭上眼睛。不要求必须把文字念出来；只要明显做出闭眼动作即可判正确。若只是念出文字但没有做动作，不应判为完全正确。"
        },
        "language_3step_action": {
            "name": "动作指令", 
            "desc": "按指令举手、握拳并把手放到胸前",
            "eval_hint": "判断患者是否完成了适合胸部以上摄像头观察的三步动作：举起右手（若说明右手不便，改用左手也可）、握成拳头、把手放到胸前。若回答明确描述自己已完成这些动作，可判为正确；若只是笼统地说'好了'、'做完了'而没有动作信息，最多算部分完成，不应直接判为完全正确。"
        },
        "language_writing_sentence": {
            "name": "说句子",
            "desc": "口头说一句完整的句子",
            "eval_hint": "判断患者是否口头说出了一个完整的句子。句子需包含主语和谓语，有实际含义（例如『今天天气真好』『我吃了早饭』）。只说单个词语、短语、感叹词，或无意义/不通顺的内容算错误。患者复述例句『今天天气真好』也算正确。"
        },
        # 构图
        "copy_pentagons": {
            "name": "临摹", 
            "desc": "临摹五边形图形",
            "eval_hint": "判断临摹图形是否包含两个五边形且有交叠。形状近似即可，不要求完美"
        },
    }
    
    _llm: ChatOpenAI = PrivateAttr()
    
    def __init__(
        self,
        use_local: bool = False,
        llm_instance = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        if use_local and llm_instance:
            print("[AnswerEvalTool] 🚀 使用传入的本地模型")
            self._llm = llm_instance
        elif use_local:
            print("[AnswerEvalTool] 🚀 使用模型池（7B）")
            from src.llm.model_pool import get_pooled_llm
            self._llm = get_pooled_llm(pool_key='eval_long')
        else:
            if os.getenv("ARK_API_KEY"):
                print("[AnswerEvalTool] � 使用火山引擎 (Doubao)")
                self._llm = get_volcengine_chat_openai(
                    model=os.getenv("ANSWER_EVAL_MODEL", "doubao-seed-2-0-lite-260215"),
                    temperature=0.05,
                    timeout=20,
                    max_retries=1,
                )
            else:
                print("[AnswerEvalTool] 🔵 使用 SiliconFlow")
                self._llm = get_siliconflow_chat_openai(
                    model=os.getenv("ANSWER_EVAL_MODEL", "Qwen/Qwen2.5-32B-Instruct"),
                    temperature=0.05,
                    timeout=20,
                    max_retries=1,
                )
    
    def _extract_response(self, response) -> str:
        """从 LLM 响应中提取文本"""
        if hasattr(response, 'content'):
            return response.content
        return str(response)
    
    def _parse_truncated_json(self, content: str) -> Dict:
        """从被截断的 JSON 中提取可用字段"""
        result = {}
        
        # 提取 is_correct
        is_correct_match = re.search(r'"is_correct"\s*:\s*(true|false)', content, re.IGNORECASE)
        if is_correct_match:
            result["is_correct"] = is_correct_match.group(1).lower() == "true"
        
        # 提取 quality_level
        quality_match = re.search(r'"quality_level"\s*:\s*"(\w+)"', content)
        if quality_match:
            result["quality_level"] = quality_match.group(1)
        else:
            quality_match = re.search(r'"quality"\s*:\s*"(\w+)"', content)
            if quality_match:
                result["quality"] = quality_match.group(1)
        
        # 提取 cognitive_performance
        cognitive_match = re.search(r'"cognitive_performance"\s*:\s*"([^"]+)"', content)
        if cognitive_match:
            result["cognitive_performance"] = cognitive_match.group(1)
        
        # 提取 is_complete
        complete_match = re.search(r'"is_complete"\s*:\s*(true|false)', content, re.IGNORECASE)
        if complete_match:
            result["is_complete"] = complete_match.group(1).lower() == "true"
        
        # 提取 evaluation_detail
        detail_match = re.search(r'"evaluation_detail"\s*:\s*"([^"]*)', content)
        if detail_match:
            result["evaluation_detail"] = detail_match.group(1)
        
        # 提取 confidence
        confidence_match = re.search(r'"confidence"\s*:\s*([\d.]+)', content)
        if confidence_match:
            try:
                result["confidence"] = float(confidence_match.group(1))
            except:
                pass

        # 提取 raw_score / raw_max_score（多分项任务可选）
        raw_score_match = re.search(r'"raw_score"\s*:\s*(\d+)', content)
        if raw_score_match:
            try:
                result["raw_score"] = int(raw_score_match.group(1))
            except (TypeError, ValueError):
                pass
        raw_max_match = re.search(r'"raw_max_score"\s*:\s*(\d+)', content)
        if raw_max_match:
            try:
                result["raw_max_score"] = int(raw_max_match.group(1))
            except (TypeError, ValueError):
                pass
 
        return result

    @staticmethod
    def _normalize_chinese_chars(text: str) -> list[str]:
        return re.findall(r"[一-鿿]", str(text or ""))

    @staticmethod
    def _char_pinyin(char: str) -> str:
        if lazy_pinyin is None:
            return char
        try:
            pinyin = lazy_pinyin(char, errors="ignore", strict=False)
            return pinyin[0] if pinyin else char
        except Exception:
            return char

    def _evaluate_reverse_phrase_homophone(self, answer: str) -> Dict[str, Any]:
        expected_chars = ["安", "平", "入", "出", "祝"]
        answer_chars = self._normalize_chinese_chars(answer)
        answer_pinyin = [self._char_pinyin(ch) for ch in answer_chars]
        expected_pinyin = [self._char_pinyin(ch) for ch in expected_chars]

        raw_score = 0
        position_details = []
        for index, expected_py in enumerate(expected_pinyin):
            actual_char = answer_chars[index] if index < len(answer_chars) else ""
            actual_py = answer_pinyin[index] if index < len(answer_pinyin) else ""
            matched = bool(actual_py and actual_py == expected_py)
            raw_score += 1 if matched else 0
            position_details.append({
                "position": index + 1,
                "expected": expected_chars[index],
                "expected_pinyin": expected_py,
                "actual": actual_char,
                "actual_pinyin": actual_py,
                "matched": matched,
            })

        if raw_score == 5:
            quality = "excellent"
        elif raw_score >= 3:
            quality = "good"
        elif raw_score >= 1:
            quality = "fair"
        else:
            quality = "poor"

        return {
            "is_correct": raw_score == 5,
            "quality_level": quality,
            "cognitive_performance": "正常" if raw_score >= 3 else "轻度异常" if raw_score >= 1 else "中度异常",
            "is_complete": raw_score == 5,
            "evaluation_detail": f"倒序词组同音容错计分：{raw_score}/5；逐位={position_details}",
            "confidence": 1.0,
            "raw_score": raw_score,
            "raw_max_score": 5,
        }

    def _run(
        self,
        question: str,
        answer: str = "",
        task_id: str = "",
        expected_answer: Optional[str] = None,
        patient_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """执行回答评估"""
        
        start_time = time.time()
        logger = ToolLogger("AnswerEvalTool")
        logger.start(
            task_id=task_id or "general",
            question=question[:60] + ("..." if len(question) > 60 else ""),
            answer=answer[:60] + ("..." if len(answer) > 60 else ""),
            expected_answer=(str(expected_answer)[:50] + "...") if expected_answer and len(str(expected_answer)) > 50 else (expected_answer or "N/A"),
        )
        
        # 容错：无回答
        if not answer or not answer.strip():
            result = json.dumps({
                "is_correct": False,
                "quality_level": "poor",
                "cognitive_performance": "中度异常",
                "is_complete": False,
                "evaluation_detail": "患者未作回答",
                "confidence": 1.0
            }, ensure_ascii=False)
            logger.log("无回答，直接返回默认评估", level="warn")
            logger.end(is_correct=False, quality="poor", detail="患者未作回答")
            return result

        if task_id == "attention_reverse_phrase":
            homophone_result = self._evaluate_reverse_phrase_homophone(answer)
            logger.log("倒着说词组使用同音字规则直接计分", score=f"{homophone_result['raw_score']}/5")
            logger.end(
                is_correct=homophone_result["is_correct"],
                quality=homophone_result["quality_level"],
                cognitive=homophone_result["cognitive_performance"],
                confidence=homophone_result["confidence"],
            )
            return json.dumps(homophone_result, ensure_ascii=False)

        # 获取任务信息
        task_info = self.TASK_INFO.get(task_id, {"name": task_id or "通用", "desc": "认知评估", "eval_hint": "判断回答是否符合问题要求"})
        
        # 获取实时上下文（用于定向力评估）
        context = get_realtime_context()
        time_info = context['time']
        location = context['location']
        
        # 🔥 v2: 详细的系统提示，减少错判
        system_prompt = """你是一位资深神经内科医生，正在对老年患者进行 MMSE（简易精神状态检查）认知筛查。
你的任务是**严格评估**患者的回答是否正确。

## 核心原则
1. **只评估回答的事实正确性**，不要被患者的语气、态度、礼貌用语影响判断
2. **口语化表达是正常的**：老人说"周五"等于"星期五"，说"93块"等于"93"，说"那个表"等于"手表"
3. **关注实质内容**：忽略语气词（嗯、啊、哦）、礼貌语（好的、知道了）、重复词
4. **宁可判对不要误判错**：如果回答包含正确信息但表述不标准，应判为正确
5. **闲聊/寒暄不是答题**：如果患者只是在聊天而没有回答问题，quality 为 fair

## 输出格式
只输出JSON，不要输出任何其他内容：
{"is_correct": true/false, "quality": "excellent/good/fair/poor", "raw_score": <int>, "raw_max_score": <int>}

- excellent: 完全正确，清晰准确
- good: 基本正确，有小瑕疵但核心信息对
- fair: 部分正确，或回答不完整
- poor: 完全错误，或答非所问

## raw_score 说明（可选）
如果评判标准明确要求输出分数（例如连续减法、三词记忆、三步动作这些多分项），
请在 JSON 中补上 `raw_score`（0到 raw_max_score 的整数）与 `raw_max_score`（该任务总分）。
如果该任务只需 对/错 二元判断，则可不输出 raw_score。"""

        # 构建用户提示
        user_prompt = f"## 评估任务\n任务: {task_info['name']}（{task_info['desc']}）\n医生的问题: {question}\n患者的回答: {answer}"
        
        # 添加评估提示
        eval_hint = task_info.get('eval_hint', '')
        if eval_hint:
            user_prompt += f"\n\n## 评判标准\n{eval_hint}"
        
        if expected_answer:
            user_prompt += f"\n\n## 正确答案\n{expected_answer}"
        
        # 定向力任务需要当前时间/地点信息
        if task_id.startswith("orientation"):
            location_parts = [
                location.get('province', ''),
                location.get('city', ''),
                location.get('district', ''),
                location.get('hospital', ''),
                location.get('department', ''),
                location.get('place', ''),
            ]
            location_text = " ".join(part for part in location_parts if part)
            user_prompt += f"\n\n## 当前真实信息（用于对比）"
            user_prompt += f"\n当前时间: {time_info['year']}年{time_info['month']}月{time_info['day']}日 星期{time_info['weekday']} {time_info['season']}"
            user_prompt += f"\n手填地址: {location_text}"
            user_prompt += f"\n手填楼层: {location.get('floor', '')}"
        
        if patient_profile:
            user_prompt += f"\n\n## 患者信息\n{patient_profile.get('age', '?')}岁, 受教育{patient_profile.get('education_years', '?')}年"
        
        user_prompt += "\n\n请只输出JSON结果："
        
        try:
            response = self._llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
            
            content = self._extract_response(response).strip()
            
            # 清理 Markdown
            if "```" in content:
                match = re.search(r"```(?:json)?(.*?)```", content, re.DOTALL)
                if match:
                    content = match.group(1).strip()
            
            # 解析 JSON（增加截断容错）
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    # JSON 被截断，尝试手动提取字段
                    logger.log("JSON 被截断，尝试容错提取", level="warn")
                    data = self._parse_truncated_json(content)
            else:
                # 没有找到完整的 JSON，尝试手动提取
                logger.log("未找到完整 JSON，尝试容错提取", level="warn")
                data = self._parse_truncated_json(content)
            
            # 🔥 简化的结果（只要核心字段）
            is_correct = bool(data.get("is_correct", False))
            quality = str(data.get("quality", data.get("quality_level", "fair")))
            
            # 根据 quality 推断其他字段
            quality_to_cognitive = {
                "excellent": "正常", "good": "正常", 
                "fair": "轻度异常", "poor": "中度异常"
            }
            
            result = {
                "is_correct": is_correct,
                "quality_level": quality,
                "cognitive_performance": quality_to_cognitive.get(quality, "无法判断"),
                "is_complete": is_correct,  # 简化：正确就视为完整
                "evaluation_detail": "",
                "confidence": 0.9 if is_correct else 0.8
            }

            # 🔥 透传多分项任务的 raw_score / raw_max_score（供上层 MMSE 记分使用）
            raw_score_val = data.get("raw_score")
            raw_max_val = data.get("raw_max_score")
            if raw_score_val is not None:
                try:
                    result["raw_score"] = int(raw_score_val)
                except (TypeError, ValueError):
                    pass
            if raw_max_val is not None:
                try:
                    result["raw_max_score"] = int(raw_max_val)
                except (TypeError, ValueError):
                    pass
            
            elapsed = (time.time() - start_time) * 1000
            logger.end(
                is_correct=is_correct,
                quality=quality,
                cognitive=result["cognitive_performance"],
                confidence=result["confidence"],
                elapsed_ms=f"{elapsed:.0f}",
            )
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.log(f"评估失败: {e}", level="error")
            result = json.dumps({
                "is_correct": False,
                "quality_level": "poor",
                "cognitive_performance": "无法判断",
                "is_complete": False,
                "evaluation_detail": "",
                "confidence": 0.0
            }, ensure_ascii=False)
            logger.end(is_correct=False, quality="poor", error=str(e)[:120])
            return result
    
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")
