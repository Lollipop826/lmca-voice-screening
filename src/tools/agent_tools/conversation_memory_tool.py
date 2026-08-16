"""
对话记忆管理工具 - 混合记忆策略

策略：LLM增量摘要 + 结构化Agent状态 + 近期窗口
- 旧消息 → 用快速LLM压缩为摘要（增量更新，不重复摘要）
- Agent状态 → 结构化提取（已完成任务、评分、记忆词等）
- 近期消息 → 保留最近4轮原文

输出给下游工具（QuestionGenTool等）的上下文格式：
  【前情摘要】...LLM生成的对话摘要 + 结构化状态...
  【最近聊天记录】...最近4轮原文...
"""

from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FutureTimeoutError
import json
import os
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# 摘要触发阈值：chat_history 超过此条数时开始摘要（设为2=首轮即触发，解决冷启动问题）
_SUMMARY_THRESHOLD = 2
# 保留近期原文的条数（最近2轮 = 4条，更早的压缩为摘要）
_RECENT_WINDOW = 4
# 摘要用 system prompt
_SUMMARY_SYSTEM_PROMPT = (
    "你是一个对话摘要助手。请将以下对话内容压缩为结构化的中文摘要，并提取已聊过的话题。\n"
    "严格输出JSON格式：\n"
    '{"summary": "摘要文本", "discussed_topics": ["话题1", "话题2"], "next_suggestion": {"from_topic": "当前话题", "to_topic": "过渡话题", "task_id": "任务ID", "anchor_fact": "引用的具体历史回答", "reason": "为什么这条历史事实适合过渡到该任务", "bridge_hint": "建议AI如何顺着这条历史事实自然转到该任务", "target_question": "下一轮最终要问用户的那一句完整问题"}}\n'
    "\n"
    "【summary要求】\n"
    "摘要应分段涵盖以下内容（有则写，无则略）：\n"
    "1. **患者基本情况**：姓名、年龄、居住地、家庭成员等\n"
    "2. **生活信息**：日常作息、饮食习惯、兴趣爱好、健康状况等\n"
    "3. **认知评估表现**：对定向力问题(年份/季节/日期/星期/地点等)的回答是否正确，"
    "计算题表现，记忆词回忆情况，命名/复述/阅读等任务表现\n"
    "4. **情绪与配合**：患者整体情绪、配合程度、是否有抵触\n"
    "5. **AI已问过的关键问题**（避免下游重复提问）\n"
    "\n"
    "写作要求：\n"
    "- 用第三人称叙述，summary尽量控制在120-220字内，默认不要超过220字\n"
    "- 保留患者的原始回答中的关键信息（具体数字、地名、人名）\n"
    "- 删除寒暄、重复、语气词等冗余内容，优先保留关键事实、认知表现和未完成任务线索\n"
    "- 如果已有之前的摘要，将新对话内容合并进去，保持连贯\n"
    "\n"
    "【discussed_topics要求】\n"
    "提取对话中已经聊过的话题类别（合并同类），例如：\n"
    '  ["饮食(早饭豆浆油条、晚饭西红柿炒蛋)", "当前地点(北京海淀区某医院)", "兴趣爱好(看书、钓鱼)"]\n'
    "每个话题用简短类别名+括号内具体细节表示。\n"
    "如果已有之前的话题列表，合并去重后输出完整列表。\n"
    "\n"
    "【next_suggestion要求】\n"
    "如果输入里要求顺带选择下一任务，则输出 next_suggestion；否则可省略该字段。\n"
    "next_suggestion 不能只写抽象话题，必须尽量绑定患者刚才或之前说过的具体事实，"
    "例如‘今天和老朋友吃饭’‘现在在北京海淀区这个地方’‘心情挺开心’等。bridge_hint 要能直接指导AI如何顺着这条事实转入任务。"
    "target_question 是下一轮最后真正要问用户的那一句完整自然问句；必须只问一件事、和 task_id 严格一致、不能泄露答案。"
    "如果 task_id 是 orientation_place_province_city / orientation_place_district / orientation_place_location_floor，"
    "target_question 必须问对方现在这个地方、当前所在的位置，并且锚定系统记录的手填地址；禁止改写成老家、住址、常住地、平时住哪。"
)


class ConversationMemoryTool:
    """
    对话记忆管理工具

    使用方法：
        memory = ConversationMemoryTool()
        # 每轮对话后更新
        memory.update(chat_history)
        # 获取上下文给 QuestionGenTool
        ctx = memory.get_context(chat_history, agent_state)
        # ctx['summary']  → 摘要字符串（含结构化状态）
        # ctx['recent']   → 最近N条原文 List[Dict]
        # 新会话时重置
        memory.reset()
    """

    def __init__(self, use_local: bool = False):
        self._summary: str = ""
        self._discussed_topics: List[str] = []  # 🔥 LLM提取的已聊话题列表
        self._summarized_up_to: int = 0
        self._llm: Optional[ChatOpenAI] = None
        self._use_local = use_local
        # 🧠 持久背景（患者跨会话记忆卡）：由会话管理方显式设置/清除，
        # 不参与增量摘要重写，也不随 reset() 清除（reset 只清本会话摘要状态）。
        self._persistent_background: str = ""
        self._turn_background: str = ""
        try:
            self._summary_char_limit = max(80, int(os.getenv("MEMORY_SUMMARY_CHAR_LIMIT", "220")))
        except ValueError:
            self._summary_char_limit = 220
        self._state_version: int = 0
        self._last_summary_time: float = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="MemorySummary")
        self._pending_future: Optional[Future] = None
        # 🔥 摘要快照（供前端可视化）
        self._snapshot_history: List[Dict[str, Any]] = []
        self._last_update_elapsed: float = 0
        self._last_update_trigger_time: float = 0  # 提交时间戳
        # 🔥 任务预建议（摘要LLM顺带生成，省一次LLM调用）
        self._suggested_next: Optional[Dict[str, str]] = None
        self._pending_task_context: Optional[Dict[str, Any]] = None
        print("[MemoryTool] ✅ 对话记忆工具初始化完成（异步模式）")

    def _limit_summary_text(self, text: Optional[str]) -> str:
        import re

        summary = (text or "").strip()
        if not summary:
            return ""

        limit = max(80, int(getattr(self, "_summary_char_limit", 220) or 220))
        if len(summary) <= limit:
            return summary

        pieces = [seg.strip() for seg in re.split(r'(?<=[。！？；])|\n+', summary) if seg and seg.strip()]
        clipped = ""
        for piece in pieces:
            if len(clipped) + len(piece) > limit:
                break
            clipped += piece
        if clipped and len(clipped) >= min(120, limit):
            return clipped.strip()
        return summary[:limit].strip()

    def _get_llm(self) -> ChatOpenAI:
        """延迟初始化摘要LLM（用快速小模型）"""
        if self._llm is None:
            if os.getenv("ARK_API_KEY"):
                from src.llm.http_client_pool import get_volcengine_chat_openai
                model = os.getenv("QUESTION_GEN_FAST_MODEL", "doubao-seed-2-0-mini-260215")
                self._llm = get_volcengine_chat_openai(
                    model=model,
                    temperature=0.1,
                    max_tokens=800,
                    timeout=15,
                    max_retries=1,
                    disable_thinking=True,
                )
                print(f"[MemoryTool] 🌋 摘要LLM: {model}")
            else:
                from src.llm.http_client_pool import get_siliconflow_chat_openai
                model = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct")
                self._llm = get_siliconflow_chat_openai(
                    model=model,
                    temperature=0.1,
                    max_tokens=800,
                    timeout=15,
                    max_retries=1,
                )
                print(f"[MemoryTool] 🔷 摘要LLM: {model}")
        return self._llm

    def update(self, chat_history: List[Dict[str, str]], task_context: Optional[Dict[str, Any]] = None) -> None:
        """
        增量更新对话摘要（可选：同时生成下一任务建议）。

        当 chat_history 超过阈值时，对 [_summarized_up_to : -_RECENT_WINDOW] 
        区间的消息调用LLM进行摘要，追加到已有摘要上。
        如果提供 task_context，还会在同一次 LLM 调用中建议下一个任务。
        """
        state_version = self._state_version

        # 合并传入的 task_context 和 pending 的
        task_ctx = task_context or self._pending_task_context
        self._pending_task_context = None  # 用完清空
        has_task_selection = bool(task_ctx and task_ctx.get('candidates'))
        total = len(chat_history)
        if total <= _SUMMARY_THRESHOLD and not has_task_selection:
            return  # 消息还不够多，不需要摘要

        # 需要摘要的区间：从上次摘要位置 到 (总数 - 近期窗口)
        summary_end = max(0, total - _RECENT_WINDOW)
        has_summary_work = summary_end > self._summarized_up_to
        if not has_summary_work and not has_task_selection:
            return  # 没有新的需要摘要的消息

        if self._summary:
            self._summary = self._limit_summary_text(self._summary)

        # 提取需要新摘要的消息
        if has_summary_work:
            new_messages = chat_history[self._summarized_up_to:summary_end]
        else:
            new_messages = chat_history[-_RECENT_WINDOW:]
        if not new_messages:
            return

        # 构建待摘要文本
        lines = []
        for msg in new_messages:
            role = "AI" if msg.get("role") == "assistant" else "患者"
            lines.append(f"{role}: {msg.get('content', '')}")
        new_text = "\n".join(lines)

        # 构建LLM请求
        if has_summary_work and self._summary:
            topics_str = json.dumps(self._discussed_topics, ensure_ascii=False) if self._discussed_topics else "[]"
            user_content = (
                f"【已有摘要】\n{self._summary}\n\n"
                f"【已提取的话题列表】\n{topics_str}\n\n"
                f"【新增对话（请合并到摘要和话题列表中）】\n{new_text}"
            )
        elif has_summary_work:
            user_content = f"【对话内容】\n{new_text}"
        else:
            topics_str = json.dumps(self._discussed_topics, ensure_ascii=False) if self._discussed_topics else "[]"
            if self._summary:
                user_content = (
                    f"【已有摘要】\n{self._summary}\n\n"
                    f"【已提取的话题列表】\n{topics_str}\n\n"
                    f"【最近对话（仅供选择下一任务，不更新摘要窗口）】\n{new_text}"
                )
            else:
                user_content = f"【最近对话（仅供选择下一任务，不更新摘要窗口）】\n{new_text}"

        summary_floor = max(80, self._summary_char_limit - 80)
        user_content += (
            f"\n\n【摘要长度限制】summary请尽量控制在{summary_floor}-{self._summary_char_limit}字内；"
            f"如果信息较多，优先保留关键事实、认知表现和未完成任务线索，务必不要超过{self._summary_char_limit}字。"
        )

        # 🔥 如果有任务上下文，追加任务选择指令
        task_selection_prompt = ""
        if task_ctx:
            candidates = task_ctx.get('candidates', [])
            task_descs = task_ctx.get('task_descriptions', {})
            used_topics = task_ctx.get('used_topics', [])
            last_topic = task_ctx.get('last_topic', '')
            if candidates:
                task_list = "、".join([f"{tid}({task_descs.get(tid, tid)})" for tid in candidates])
                used_str = "、".join(used_topics) if used_topics else "无"
                task_selection_prompt = (
                    f"\n\n【任务选择（重要！请一并输出）】\n"
                    f"请根据对话进展，从以下待评估任务中选一个最适合自然过渡的：\n"
                    f"待选任务：{task_list}\n"
                    f"已聊话题（禁止重复）：{used_str}\n"
                    f"上一个话题：{last_topic or '无'}\n"
                    f"选择原则：优先利用患者最近明确说过的具体事实来过渡，而不是只写抽象话题标签。"
                    f"例如患者说‘今天和老朋友一起吃了顿饭’，可以先顺着这条事实回应，再自然转到时间/日期/季节；"
                    f"患者提到‘在北京海淀区这个医院/地方’，可以顺到当前这个地方的区县或具体地点。\n"
                    f"请额外给出：\n"
                    f"1. anchor_fact：引用哪条具体历史回答/事实\n"
                    f"2. reason：为什么这条事实适合过渡到该任务\n"
                    f"3. bridge_hint：给AI的一句具体过渡提示，要求能直接用于生成问题\n"
                    f"4. target_question：下一轮真正要问用户的那一句完整自然问句；如果不适合提前定死，可留空字符串\n"
                    f"如果 task_id 是 orientation_place_province_city / orientation_place_district / orientation_place_location_floor，target_question 必须问对方现在这个地方、当前所在的位置，且以系统记录的手填地址为准；禁止问老家、住址、常住地或平时住哪。\n"
                    f'在JSON中增加 "next_suggestion": {{"from_topic": "当前话题", "to_topic": "过渡话题", "task_id": "任务ID", "anchor_fact": "引用的具体历史回答", "reason": "选择原因", "bridge_hint": "具体过渡提示", "target_question": "完整自然问句"}}'
                )
                user_content += task_selection_prompt

        try:
            t0 = time.time()
            llm = self._get_llm()
            response = llm.invoke([
                SystemMessage(content=_SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ])
            raw = response.content.strip()
            # 解析JSON响应，提取 summary 和 discussed_topics
            parsed = self._parse_summary_response(raw)
            summary_text = self._limit_summary_text(parsed.get('summary', raw))
            if state_version != self._state_version:
                return
            # 🔥 提取任务建议
            next_sug = parsed.get('next_suggestion')
            if next_sug and isinstance(next_sug, dict) and next_sug.get('task_id'):
                self._suggested_next = next_sug
                _target_question = (next_sug.get('target_question') or '').strip()
                print(f"[MemoryTool] 🎯 任务预建议: {next_sug.get('task_id')} "
                      f"({next_sug.get('from_topic', '')}→{next_sug.get('to_topic', '')})"
                      f"{f' target={_target_question[:24]}...' if _target_question else ''}")
            elapsed = time.time() - t0
            self._last_update_elapsed = elapsed
            if has_summary_work:
                self._summary = summary_text
                new_topics = parsed.get('discussed_topics', [])
                if new_topics:
                    self._discussed_topics = new_topics
                self._summarized_up_to = summary_end
                self._last_summary_time = time.time()
                # 🔥 记录快照
                trigger_ts = self._last_update_trigger_time or t0
                self._snapshot_history.append({
                    'time': time.strftime('%H:%M:%S'),
                    'trigger_to_done': round(time.time() - trigger_ts, 2),
                    'llm_elapsed': round(elapsed, 2),
                    'summary_len': len(self._summary),
                    'summary': self._summary,
                    'topics': list(self._discussed_topics),
                    'msg_range': f'0-{summary_end}',
                })
                print(
                    f"[MemoryTool] 🧠 摘要更新完成: "
                    f"覆盖消息 [0:{summary_end}], "
                    f"摘要长度={len(self._summary)}字, "
                    f"已聊话题={self._discussed_topics}, "
                    f"耗时={elapsed:.2f}s"
                )
            else:
                trigger_ts = self._last_update_trigger_time or t0
                self._snapshot_history.append({
                    'time': time.strftime('%H:%M:%S'),
                    'trigger_to_done': round(time.time() - trigger_ts, 2),
                    'llm_elapsed': round(elapsed, 2),
                    'summary_len': len(self._summary),
                    'summary': self._summary,
                    'topics': list(self._discussed_topics),
                    'msg_range': 'task-only',
                })
                print(f"[MemoryTool] ⚡ 任务预选完成（未推进摘要窗口，耗时={elapsed:.2f}s）")
        except Exception as e:
            print(f"[MemoryTool] ⚠️ 摘要生成失败（降级为无摘要）: {type(e).__name__}")

    def _parse_summary_response(self, raw: str) -> dict:
        """解析 LLM 返回的 JSON 响应，容错处理（支持嵌套对象如 next_suggestion）。"""
        import re
        # 清理 markdown 代码块
        cleaned = re.sub(r'```json\s*|\s*```', '', raw).strip()
        # 尝试直接解析
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and 'summary' in data:
                return data
        except json.JSONDecodeError:
            pass
        # 尝试提取最外层 {} 块（贪心匹配，支持嵌套）
        brace_depth = 0
        start_idx = None
        for i, ch in enumerate(cleaned):
            if ch == '{':
                if brace_depth == 0:
                    start_idx = i
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if brace_depth == 0 and start_idx is not None:
                    try:
                        data = json.loads(cleaned[start_idx:i+1])
                        if isinstance(data, dict) and 'summary' in data:
                            return data
                    except json.JSONDecodeError:
                        pass
                    start_idx = None
        # 兜底：把整个响应当作纯文本摘要
        print(f"[MemoryTool] ⚠️ JSON解析失败，降级为纯文本摘要: {cleaned[:100]}")
        return {'summary': raw, 'discussed_topics': self._discussed_topics}

    def get_discussed_topics(self) -> List[str]:
        """返回 LLM 提取的已聊话题列表（供话题选择去重）。"""
        self._wait_pending()
        return list(self._discussed_topics)

    def get_discussed_topics_snapshot(self) -> List[str]:
        """非阻塞返回当前已聊话题快照。"""
        return list(self._discussed_topics)

    def get_discussed_topics_str(self) -> str:
        """返回已聊话题的逗号分隔字符串，用于直接注入 prompt。"""
        topics = self.get_discussed_topics()
        return "、".join(topics) if topics else "无"

    def get_discussed_topics_str_snapshot(self) -> str:
        """非阻塞返回已聊话题字符串快照。"""
        topics = self.get_discussed_topics_snapshot()
        return "、".join(topics) if topics else "无"

    def update_async(self, chat_history: List[Dict[str, str]], task_context: Optional[Dict[str, Any]] = None) -> None:
        """
        异步更新摘要+任务预建议：提交到后台线程，不阻塞主流程。
        在 process_turn 返回响应后调用，下一轮 get_context() 时自动等待完成。
        """
        # 快速判断：不需要摘要时直接跳过，不提交线程
        total = len(chat_history)
        has_task_selection = bool(task_context and task_context.get('candidates'))
        if total <= _SUMMARY_THRESHOLD and not has_task_selection:
            print(f"[MemoryTool] ⏭️ 消息数={total} ≤ 阈值{_SUMMARY_THRESHOLD}，跳过摘要")
            return
        summary_end = max(0, total - _RECENT_WINDOW)
        has_summary_work = summary_end > self._summarized_up_to
        if not has_summary_work and not has_task_selection:
            print(f"[MemoryTool] ⏭️ 无新消息需摘要（已覆盖到[0:{self._summarized_up_to}]）")
            return

        # 等待上一次异步摘要完成（防止并发写 _summary）
        self._wait_pending()

        # 提交到后台线程
        new_msg_count = max(0, summary_end - self._summarized_up_to)
        self._last_update_trigger_time = time.time()
        self._pending_task_context = task_context  # 保存给 update() 使用
        self._pending_future = self._executor.submit(self.update, list(chat_history))
        if has_summary_work:
            has_task = " + 任务预选" if has_task_selection else ""
            print(f"[MemoryTool] 🚀 异步摘要{has_task}已提交: {new_msg_count}条新消息待压缩（后台执行，不阻塞响应）")
        else:
            print(f"[MemoryTool] 🚀 异步任务预选已提交（无需推进摘要窗口，后台执行，不阻塞响应）")

    def _wait_pending(self, timeout: float = 10, block: bool = True) -> bool:
        """等待挂起的异步摘要完成。"""
        future = self._pending_future
        if future is None:
            return True

        if future.done():
            try:
                future.result()
            except Exception as e:
                print(f"[MemoryTool] ⚠️ 后台摘要失败: {type(e).__name__}")
            self._pending_future = None
            return True

        if not block:
            print("[MemoryTool] ⏭️ 后台摘要仍在执行，get_context 先使用当前摘要快照")
            return False

        t0 = time.time()
        print(f"[MemoryTool] ⏳ 等待后台摘要完成...")
        try:
            future.result(timeout=timeout)
            print(f"[MemoryTool] ✅ 异步摘要已就绪（等待了 {time.time()-t0:.2f}s）")
        except FutureTimeoutError:
            print(f"[MemoryTool] ⚠️ 等待异步摘要超时({time.time()-t0:.2f}s)，继续保留后台任务")
            return False
        except Exception as e:
            print(f"[MemoryTool] ⚠️ 等待异步摘要失败({time.time()-t0:.2f}s): {type(e).__name__}")
        finally:
            if future.done():
                self._pending_future = None

        return future.done()

    def set_persistent_background(self, text: str) -> None:
        """设置持久背景（患者跨会话记忆卡）。

        会拼在 get_context() 摘要的最前面，且不被增量摘要 LLM 重写。
        传空字符串即清除。
        """
        self._persistent_background = (text or "").strip()
        if self._persistent_background:
            print(f"[MemoryTool] 🗂️ 已注入持久背景（患者记忆卡，{len(self._persistent_background)}字）")

    def set_turn_background(self, text: str) -> None:
        """设置只供当前一轮 Agent 使用的长期记忆。"""
        self._turn_background = (text or "").strip()

    def get_context(
        self,
        chat_history: List[Dict[str, str]],
        agent_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        获取完整上下文（供 QuestionGenTool / ComfortTool 使用）。

        Returns:
            {
                'summary': str,          # 摘要 + 结构化状态
                'recent': List[Dict],    # 最近N条原文
            }
        """
        self._wait_pending(block=False)
        if self._summary:
            self._summary = self._limit_summary_text(self._summary)
        
        has_summary = "有" if self._summary else "无"
        print(f"[MemoryTool] 📦 get_context: 摘要={has_summary}({len(self._summary)}字), 历史={len(chat_history)}条")
        
        # 1. 只保留最近2轮原话，每条最多500字。
        recent = []
        for message in chat_history[-_RECENT_WINDOW:]:
            item = dict(message)
            item["content"] = str(item.get("content") or "")[:500]
            recent.append(item)

        # 2. 组合摘要：持久背景（跨会话记忆卡）→ 本会话摘要 → 结构化状态
        parts = []
        if self._persistent_background:
            parts.append(self._persistent_background)
        if self._turn_background:
            parts.append(self._turn_background)
        if self._summary:
            parts.append(self._summary)

        # 3. 结构化Agent状态
        if agent_state:
            state_text = self._build_structured_state(agent_state)
            if state_text:
                parts.append(state_text)

        summary = "\n".join(parts) if parts else ""

        return {
            'summary': summary,
            'recent': recent,
        }

    @staticmethod
    def _build_structured_state(agent_state: Dict[str, Any]) -> str:
        """从Agent状态提取结构化摘要"""
        lines = []

        # 已完成的任务
        task_done = agent_state.get('task_done')
        if task_done:
            # 将内部task_id映射为可读名称
            task_labels = {
                "orientation_time_year": "年份",
                "orientation_time_season": "季节",
                "orientation_time_month_date": "月份/日期",
                "orientation_time_weekday": "星期几",
                "orientation_place_province_city": "省/市",
                "orientation_place_district": "区/县",
                "orientation_place_location_floor": "地点/楼层",
                "registration_3words": "三词记忆登记",
                "attention_calc_life_math": "100连减7计算",
                "recall_3words": "三词延迟回忆",
                "language_naming_watch": "命名(手表)",
                "language_naming_pencil": "命名(铅笔)",
                "language_repetition_sentence": "复述短句",
                "language_reading_close_eyes": "阅读理解(闭眼)",
                "language_3step_action": "三步指令",
                "language_writing_sentence": "说一句话",
                "copy_pentagons": "临摹五边形",
            }
            done_names = [task_labels.get(t, t) for t in task_done]
            lines.append(f"已完成评估：{', '.join(done_names)}")

        # 记忆词
        memory_words = agent_state.get('memory_words')
        if memory_words:
            lines.append(f"登记的记忆词：{memory_words}")

        # 当前MMSE总分
        mmse_total = agent_state.get('mmse_total_score')
        if mmse_total is not None:
            lines.append(f"当前MMSE得分：{mmse_total}/35")

        # 患者兴趣/钩子
        persona_hooks = agent_state.get('persona_hooks')
        if persona_hooks:
            lines.append(f"患者兴趣/习惯：{', '.join(persona_hooks)}")

        return "\n".join(lines)

    def reset(self) -> None:
        """重置记忆（新会话时调用）"""
        self._state_version += 1
        self._wait_pending()  # 等待挂起的摘要完成再重置
        self._summary = ""
        self._discussed_topics = []
        self._summarized_up_to = 0
        self._last_summary_time = 0
        self._pending_future = None
        self._suggested_next = None
        self._pending_task_context = None
        self._turn_background = ""
        print("[MemoryTool] 🔄 记忆已重置")

    def get_and_clear_suggestion(self) -> Optional[Dict[str, str]]:
        """
        获取并清除任务预建议（原子操作）。
        返回 {"from_topic": "...", "to_topic": "...", "task_id": "..."} 或 None。
        不等待异步摘要完成（非阻塞），只取已就绪的结果。
        """
        sug = self._suggested_next
        if sug:
            self._suggested_next = None
            return sug
        return None

    def get_snapshot(self) -> Dict[str, Any]:
        """返回当前摘要快照（供前端可视化）"""
        is_running = (self._pending_future is not None and not self._pending_future.done())
        return {
            'status': 'running' if is_running else 'idle',
            'current_summary': self._summary,
            'current_topics': list(self._discussed_topics),
            'summary_len': len(self._summary),
            'summarized_up_to': self._summarized_up_to,
            'last_elapsed': round(self._last_update_elapsed, 2),
            'persistent_background': self._persistent_background,
            'turn_background': self._turn_background,
            'history': self._snapshot_history[-10:],  # 最近10次
        }

    @property
    def has_summary(self) -> bool:
        return bool(self._summary)

    @property
    def summary_text(self) -> str:
        return self._summary
