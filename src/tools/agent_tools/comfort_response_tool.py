"""
情感安慰工具 - 供Agent调用

在检测到患者抵抗情绪后，生成温和、专业的安慰回复
"""

from __future__ import annotations

from typing import Optional, Type, ClassVar, Dict, List
import os
import json

from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from src.llm.http_client_pool import get_chat_openai, get_siliconflow_chat_openai


class ComfortResponseToolArgs(BaseModel):
    """情感安慰工具参数"""
    
    resistance_category: str = Field(
        ..., 
        description="抵抗情绪类别：refusal（拒绝）/avoidance（回避）/hostility（敌意）/fatigue（疲惫）"
    )
    patient_answer: Optional[str] = Field(
        default=None,
        description="患者的原始回答（用于更个性化的安慰）"
    )
    resistance_reason: Optional[str] = Field(
        default=None,
        description="抵抗情绪的检测理由"
    )
    patient_age: Optional[int] = Field(default=None, description="患者年龄")
    patient_name: Optional[str] = Field(default=None, description="患者姓名")
    patient_gender: Optional[str] = Field(default=None, description="患者性别（男/女）")
    chat_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="最近的聊天记录（用于上下文感知）"
    )
    used_topics: Optional[List[str]] = Field(
        default=None, 
        description="已使用过的闲聊话题列表（用于去重）"
    )


class ComfortResponseResult(BaseModel):
    """安慰回复结果"""
    
    success: bool = Field(..., description="是否成功生成安慰回复")
    comfort_message: str = Field(..., description="安慰性回复内容")
    tone: str = Field(default="gentle", description="回复语气：gentle（温和）/empathetic（共情）/supportive（支持性）")
    suggestion: Optional[str] = Field(default=None, description="可选的建议（如建议休息等）")
    selected_topic: Optional[str] = Field(default=None, description="本次选用的闲聊话题（供Agent记录）")


class ComfortResponseTool(BaseTool):
    """
    情感安慰工具
    
    在检测到患者抵抗情绪后，生成温和、专业的安慰回复。
    安慰内容纯粹是情感支持，不涉及具体的评估问题。
    """
    
    name: str = "comfort_response_tool"
    description: str = (
        "生成针对患者抵抗情绪的安慰性回复。"
        "当检测到患者有拒绝、回避、敌意或疲惫等负面情绪时使用。"
        "输入参数：resistance_category（情绪类别）、patient_answer（患者回答）。"
        "返回：温和的安慰话语，纯粹情感支持，不涉及评估问题。"
        "适用场景：在resistance_detection_tool检测到抵抗情绪后立即使用。"
    )
    
    args_schema: Type[BaseModel] = ComfortResponseToolArgs
    
    _llm: ChatOpenAI = PrivateAttr()
    _system_prompt: str = PrivateAttr()
    
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.8,  # 较高温度，使回复更自然、温暖
        use_local: bool = False,  # 已废弃，强制使用 API
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        if base_url or api_key:
            print("[ComfortTool] 使用显式兼容 API 配置")
            self._llm = get_siliconflow_chat_openai(
                model=model or os.getenv("COMFORT_TOOL_MODEL", "Qwen/Qwen2.5-72B-Instruct"),
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                max_tokens=150,
                timeout=20,
                max_retries=1,
            )
        else:
            selected_model = model or os.getenv("COMFORT_TOOL_MODEL") or None
            if selected_model is None and not os.getenv("DASHSCOPE_API_KEY"):
                selected_model = (
                    "doubao-seed-2-0-lite-260215"
                    if os.getenv("ARK_API_KEY")
                    else "Qwen/Qwen2.5-72B-Instruct"
                )
            print("[ComfortTool] 使用统一 LLM 路由")
            self._llm = get_chat_openai(
                model=selected_model,
                temperature=temperature,
                max_tokens=150,
                timeout=20,
                max_retries=1,
            )
        
        # 🎯 针对不同抵抗类别的专属提示词（4类体系）
        # 每个类别独立prompt，语气要像晚辈/邻居跟老人聊天
        self._category_prompts = {
            # 0-正常配合
            "normal": (
                "老人态度配合，简单接话继续聊就行。\n\n"
                "【示例】\n"
                "老人：「好，我试试看」\n"
                "你：「成，那咱接着唠」"
            ),
            
            # 1-消极回避（合并：denial + avoidance + shame）
            "avoidance": (
                "老人不太想正面回答，可能是觉得没问题、想躲开、或者觉得丢人。\n"
                "别跟他犟，顺着说，夸两句，换个轻松的话题。\n\n"
                "【示例】\n"
                "老人：「我脑子好着呢，没病」\n"
                "你：「是是是，您这精神头多好啊。对了，早上吃的啥？」\n\n"
                "老人：「这个改天再说吧」\n"
                "你：「行，那咱先聊点别的，您今儿心情咋样？」\n\n"
                "老人：「这么简单都不会，真丢脸」\n"
                "你：「啥呀，这有啥的。我昨儿出门忘带钥匙，在门口站半天」"
            ),
            
            # 2-情绪困扰（合并：anxiety + sadness + distrust）
            "distress": (
                "老人情绪不太好，可能是紧张害怕、心情低落、或者不信任。\n"
                "给他温暖和陪伴，别急着解释，态度诚恳就行。\n\n"
                "【示例】\n"
                "老人：「这太难了，我怕答不上来」\n"
                "你：「没事儿没事儿，咱就随便唠，答不上来拉倒呗」\n\n"
                "老人：「活着也没啥意思」\n"
                "你：「您可别这么说，家里人天天惦记您呢，孩子们都盼着您好好的」\n\n"
                "老人：「你到底是干啥的」\n"
                "你：「就是来陪您说说话的，您要是不想聊，咱不聊也行」"
            ),
            
            # 3-主动对抗（合并：anger + reactance）
            "hostility": (
                "老人不高兴了，可能是烦了生气了，或者不想被管。\n"
                "赶紧认怂，别解释，把主动权给他，让他消消气。\n\n"
                "【示例】\n"
                "老人：「问什么问，烦不烦啊」\n"
                "你：「行行行，不问了不问了，是我多嘴，您歇着」\n\n"
                "老人：「我说不做就不做」\n"
                "你：「成，都听您的，您说咋整咱就咋整」\n\n"
                "老人：「轮得着你管吗」\n"
                "你：「不管不管，您自己做主，我就是陪您聊聊」"
            ),
        }
        
        # 🔥 向后兼容：旧9类标签映射到新4类
        self._category_prompts["denial"] = self._category_prompts["avoidance"]
        self._category_prompts["shame"] = self._category_prompts["avoidance"]
        self._category_prompts["anxiety"] = self._category_prompts["distress"]
        self._category_prompts["sadness"] = self._category_prompts["distress"]
        self._category_prompts["distrust"] = self._category_prompts["distress"]
        self._category_prompts["anger"] = self._category_prompts["hostility"]
        self._category_prompts["reactance"] = self._category_prompts["hostility"]
        # fatigue 兜底到 avoidance
        self._category_prompts["fatigue"] = self._category_prompts["avoidance"]
        
        # 通用基础提示词
        self._base_prompt = (
            "你是患者的晚辈/朋友，正在陪老人聊天。\n"
            "根据上面的示例风格，针对老人说的话生成回复。\n\n"
            "🚨 规则：\n"
            "- 必须口语化，像邻居唠嗑\n"
            "- 必须简短，1-2句话\n"
            "- 禁止说「我能理解您的感受」这种官话\n"
            "- 禁止说「医疗」「评估」「认知」等专业词\n"
            "- 只能基于患者原话或上下文已出现的信息，不得臆造人物/地点/事件\n"
            "- 如果上下文没有具体事实，只能用泛化问法（如“您平时喜欢做啥”）\n\n"
            "直接输出回复内容，不要JSON格式。"
        )
    
    # ⚡ 预设安慰话语模板（0延迟）
    # 支持新的8类分类：normal, shame, denial, fatigue, anxiety, anger, confabulation, distrust
    COMFORT_TEMPLATES: ClassVar[Dict[str, List[str]]] = {
        # 正常配合 - 不需要安慰
        'normal': [
            "好的，咱们继续。",
            "行，那咱们接着聊。",
        ],
        # 病耻感 - 感到羞耻尴尬
        'shame': [
            "没事儿，谁都有忘事的时候。",
            "这很正常，不用往心里去。",
            "别担心，咱就是随便聊聊。",
        ],
        # 否认病情 - 不承认有问题
        'denial': [
            "好好好，您说的对。咱们换个话题？",
            "行，那就不说这个了。",
            "您精神头挺好的。咱们聊点别的？",
        ],
        # 疲劳放弃
        'fatigue': [
            "累了就歇着，不着急。",
            "好，那您先休息，想聊了再叫我。",
            "行，歇会儿吧。",
        ],
        # 焦虑回避
        'anxiety': [
            "没事儿，不着急，慢慢来。",
            "别紧张，咱就是说着玩的。",
            "没关系，想不起来就算了。",
        ],
        # 愤怒对抗
        'anger': [
            "好好好，不问了。您歇会儿吧。",
            "行，那咱们先不聊了。",
            "抱歉打扰您了，您休息休息。",
        ],
        # 虚构掩饰
        'confabulation': [
            "好，我记住了。您接着说？",
            "嗯嗯，明白了。",
            "行，那咱们换个话题？",
        ],
        # 不信任
        'distrust': [
            "我就是陪您聊聊天，没别的意思。",
            "咱们就是随便说说话，您放心。",
            "我是来陪您解闷的，不是来检查的。",
        ],
        # 兼容旧类别名（回退）
        'refusal': [
            "好的，那咱们先不聊这个了。",
            "没关系，不想说就不说。",
        ],
        'hostility': [
            "好好好，不问了。",
            "行，那咱们歇会儿。",
        ],
        'avoidance': [
            "没事，不着急。",
            "好，那咱们换个轻松的。",
        ],
        'goodbye': [
            "好，那您休息吧。想聊了随时叫我。",
            "行，您歇着。回头再聊。",
        ],
        'bored': [
            "那您想干点啥？看会儿电视还是刷刷手机？",
            "要不我给您讲个笑话？",
        ],
    }

    def _build_safe_fallback(self, full_honorific: Optional[str], selected_topic: Optional[str]) -> str:
        """当回复出现臆造细节时，回退到稳健的泛化问法。"""
        topic_hint = (selected_topic or "").strip()
        if "天气" in topic_hint:
            followup = "今天天气还挺合适出门，您平时会不会出去走走？"
        elif "饮食" in topic_hint:
            followup = "您平时吃饭最喜欢吃点啥呀？"
        elif "电视" in topic_hint:
            followup = "您平时喜欢看什么节目呀？"
        elif "兴趣" in topic_hint or "爱好" in topic_hint:
            followup = "您平时最喜欢做点啥呀？"
        else:
            followup = "要不聊聊您平时最喜欢做点啥？"

        prefix = f"{full_honorific}，" if full_honorific else ""
        return f"{prefix}听您这么说我也挺高兴的。{followup}"

    def _has_ungrounded_detail(self, reply: str, source_text: str) -> bool:
        """
        检测明显的“凭空细节”。
        仅做保守拦截：命中高风险短语且上下文未出现对应线索才拦截。
        """
        text = reply or ""
        source = source_text or ""
        grounded_rules = [
            ("咱们村", ["村"]),
            ("村里", ["村"]),
            ("老槐树", ["槐树", "树"]),
            ("您家老伴", ["老伴", "爱人"]),
            ("你家老伴", ["老伴", "爱人"]),
            ("您儿子", ["儿子", "孩子"]),
            ("您女儿", ["女儿", "孩子"]),
            ("您孙子", ["孙子", "孙女"]),
        ]
        for phrase, hints in grounded_rules:
            if phrase in text and not any(h in source for h in hints):
                return True
        return False
    
    def _run(
        self,
        resistance_category: str,
        patient_answer: Optional[str] = None,
        resistance_reason: Optional[str] = None,
        patient_age: Optional[int] = None,
        patient_name: Optional[str] = None,
        patient_gender: Optional[str] = None,  # 🔥 新增：性别参数
        used_topics: Optional[List[str]] = None,  # 🔥 新增参数：已用过的闲聊话题
        chat_history: Optional[List[Dict[str, str]]] = None,  # 🔥 新增：聊天记录
        use_template: bool = False,
    ) -> str:  # type: ignore[override]
        """
        生成情感安慰回复，并自动选择最优闲聊话题
        """
        import time
        import random
        import re
        from src.utils.tool_logger import ToolLogger
        
        logger = ToolLogger("ComfortTool")
        logger.start(
            患者回答=patient_answer[:40] if patient_answer else "N/A",
            情绪类型=resistance_category,
            已用话题数=len(used_topics) if used_topics else 0
        )
        
        # 🔥 生成完整称呼（如：张爷爷）
        full_honorific = None
        if patient_name:
            try:
                _age = int(patient_age) if patient_age else 60
            except (ValueError, TypeError):
                _age = 60
            if patient_gender == '男':
                suffix = '爷爷' if _age >= 60 else '叔叔'
            else:
                suffix = '奶奶' if _age >= 60 else '阿姨'
            full_honorific = f"{patient_name}{suffix}"
        
        # 1. ⚡ 快速模式：使用预设模板
        if use_template:
            templates = self.COMFORT_TEMPLATES.get(resistance_category, self.COMFORT_TEMPLATES['refusal'])
            comfort_message = random.choice(templates)
            if full_honorific:
                comfort_message = f"{full_honorific}，" + comfort_message
            
            result = ComfortResponseResult(
                success=True,
                comfort_message=comfort_message,
                tone="gentle",
                suggestion=None,
                selected_topic=None
            )
            logger.end(
                回复=comfort_message[:50] + "..." if len(comfort_message) > 50 else comfort_message,
                选中话题="template",
            )
            return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
        
        # 2. 🔥 智能模式（优化版）：合并话题生成+安慰语生成为 1 次 LLM 调用（省 1-2 秒）
        try:
            # 构建上下文描述
            context_desc = ""
            source_text = patient_answer or ""
            if chat_history and len(chat_history) > 2:
                recent = chat_history[-6:]
                context_desc = "【最近聊天上下文】\n"
                for msg in recent:
                    role = "医生" if msg.get("role") == "assistant" else "患者"
                    content = msg.get('content', '')
                    context_desc += f"{role}: {content}\n"
                    source_text += f"\n{content}"
                context_desc += "\n"
            
            used_topics = used_topics or []
            used_topics_str = ", ".join(used_topics) if used_topics else "无"
            
            # 获取情绪策略提示
            category_prompt = self._category_prompts.get(
                resistance_category, 
                self._category_prompts.get("anger", "")
            )
            
            # 🔥 合并 prompt：一次调用同时完成安慰语生成+话题选择
            combined_prompt = (
                f"{category_prompt}\n\n"
                f"{self._base_prompt}\n\n"
                f"{context_desc}"
                f"🚨 任务（严格按顺序）：\n"
                f"1. 【最重要】先回应老人的情绪，按上面的策略示例风格安抚。\n"
                f"   - 如果是 hostility，reply 必须以认怂/道歉开头（如“行行行，不问了”“是我多嘴了”）\n"
                f"   - 如果是 avoidance，reply 必须先顺着说/夸两句\n"
                f"   - 如果是 distress，reply 必须先给温暖和陪伴\n"
                f"2. 然后再自然地转到一个轻松话题（避开已用话题：{used_topics_str}）\n"
                f"3. 话题要适合老年人（日常、天气、饮食、电视、兴趣爱好）\n"
                f"4. 严禁臆造具体经历或场景；如无明确信息，只能问泛化问题\n\n"
                f"【禁止】reply 不能跳过安抚直接转话题！必须先有情绪回应再过渡。\n\n"
                f"输出格式（严格JSON）：\n"
                f'{{"topic": "选中的话题", "reply": "先安抚再转场的回复"}}'
            )
            
            user_prompt = f"患者说：「{patient_answer or '不想回答'}」\n情绪：{resistance_category}"
            if full_honorific:
                user_prompt += f"\n🚨称呼规则：全程只能用'{full_honorific}'称呼对方，严禁使用其他任何变体"
            
            response = self._llm.invoke([
                {"role": "system", "content": combined_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            content = str(response.content if hasattr(response, "content") else response).strip()
            
            # 解析合并结果
            selected_topic = None
            comfort_msg = content
            try:
                # 尝试解析 JSON
                content_clean = re.sub(r'```json\s*|\s*```', '', content)
                json_match = re.search(r'\{[\s\S]*\}', content_clean)
                if json_match:
                    data = json.loads(json_match.group())
                    selected_topic = data.get("topic", "休息")
                    comfort_msg = data.get("reply", content)
            except:
                logger.log("JSON解析失败，使用原始回复", level="warn")
                selected_topic = "休息"
            
            # 清理
            comfort_msg = comfort_msg.strip().strip('"').strip("'")
            if comfort_msg.startswith("你回复："): comfort_msg = comfort_msg[4:]
            if self._has_ungrounded_detail(comfort_msg, source_text):
                logger.log("检测到潜在臆造细节，回退为稳健问法", level="warn")
                comfort_msg = self._build_safe_fallback(full_honorific, selected_topic)
            
            logger.step(f"✅ 选中话题: {selected_topic}")
            
            result = ComfortResponseResult(
                success=True,
                comfort_message=comfort_msg,
                tone="gentle",
                suggestion=None,
                selected_topic=selected_topic
            )
            
            logger.end(
                回复=comfort_msg[:50] + "..." if len(comfort_msg) > 50 else comfort_msg,
                选中话题=selected_topic
            )
            return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.log(f"生成失败: {e}", level="error")
            # 降级处理
            result_json = json.dumps({
                "success": True,
                "comfort_message": "没关系，咱们先歇会儿。",
                "tone": "gentle",
                "selected_topic": "rest"
            }, ensure_ascii=False)
            logger.end(回复="没关系，咱们先歇会儿。", 选中话题="rest", 状态="fallback")
            return result_json

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")
