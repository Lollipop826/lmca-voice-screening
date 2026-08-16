"""
视觉评估工具 - 使用视觉大模型评估患者动作/图形

支持视频和图片两种输入模式：
- 视频模式（推荐）：录制 5-10 秒短视频，VL 模型逐帧分析动作
- 图片模式（降级）：单帧截图分析

支持的 MMSE 视觉任务：
1. language_reading_close_eyes - 判断患者是否闭眼
2. language_3step_action - 判断患者是否执行三步动作
3. copy_pentagons - 判断患者临摹的五边形是否正确
"""

import os
import json
import re
import time
import base64
import shutil
import subprocess
import tempfile
from typing import Optional, Dict, Any

from src.llm.http_client_pool import get_shared_httpx_client


# 各视觉任务的评估 prompt（视频模式）
VISION_TASK_PROMPTS: Dict[str, Dict[str, str]] = {
    "language_reading_close_eyes": {
        "system": "你是一位神经内科医生的助手，正在通过摄像头视频辅助评估老年患者的认知能力。",
        "prompt": """请观察这段视频中的人物，判断他/她是否**闭上了眼睛**。

注意观察整个视频过程中眼部的变化：
- 视频中有明确的闭眼动作（从睁眼到闭眼） → is_correct: true, quality: "excellent"
- 视频中大部分时间眼睛闭合或半闭 → is_correct: true, quality: "good"
- 视频中眼睛始终睁开，没有闭眼动作 → is_correct: false, quality: "poor"
- 看不清人脸/无人 → is_correct: null, quality: "unknown"

只输出JSON：{"is_correct": true/false/null, "quality": "excellent/good/poor/unknown", "detail": "简短描述"}""",
    },
    "language_3step_action": {
        "system": "你是一位神经内科医生的助手，正在通过摄像头视频辅助评估老年患者的认知能力。",
        "prompt": """请观察这段视频，判断患者是否完成了适合胸部以上摄像头观察的三步动作：
1. 举起右手，让摄像头能看到（若右手不便，允许改用左手）
2. 把手握成拳头
3. 把这只手放到胸前

重点观察：
- 是否真的举起一只手，让摄像头能看到
- 是否出现明显握拳动作
- 是否随后把手放到胸前/上胸前

评分标准：
- 三个动作都完成，顺序基本正确 → is_correct: true, quality: "excellent"
- 主要动作完成，但手别不同、顺序略有偏差或画面略不清楚 → is_correct: true, quality: "good"
- 只完成了部分动作，或只口头回应未见实际操作 → is_correct: false, quality: "fair"
- 没有完成关键动作或明显做错 → is_correct: false, quality: "poor"
- 看不清、无人、看不到手部 → is_correct: null, quality: "unknown"

请务必额外给出 steps_completed，取值 0-3，表示三步里实际完成了几步。

只输出JSON：{"is_correct": true/false/null, "quality": "excellent/good/fair/poor/unknown", "steps_completed": 0-3, "detail": "简短描述"}""",
    },
    "copy_pentagons": {
        "system": "你是一位神经内科医生的助手，正在通过视频评估患者临摹的五边形图形。",
        "prompt": """请观察这段视频中患者画的图形，判断是否正确临摹了**两个相交的五边形**。

MMSE 评分标准：
- 两个五边形都是五条边，且有一个交叠区域 → is_correct: true, quality: "excellent"（1分）
- 基本能看出两个五边形且有交叠，但形状不够标准 → is_correct: true, quality: "good"（1分）
- 只画了一个图形，或两个图形没有交叠 → is_correct: false, quality: "fair"（0分）
- 完全无法辨认 → is_correct: false, quality: "poor"（0分）
- 看不到画作 → is_correct: null, quality: "unknown"

只输出JSON：{"is_correct": true/false/null, "quality": "excellent/good/fair/poor/unknown", "detail": "简短描述"}""",
    },
    "language_writing_sentence": {
        "system": "你是一位神经内科医生的助手，正在评估患者手写的句子是否符合MMSE书写任务要求。",
        "prompt": """请观察这张图片中患者手写的内容，判断是否写了一个**完整的句子**。

MMSE 书写评分标准：
- 句子必须有主语和谓语（动词），且语义通顺 → is_correct: true, quality: "excellent"（1分）
- 句子结构基本完整，但有小瑕疵（如缺少标点、字迹潦草但可辨认） → is_correct: true, quality: "good"（1分）
- 只写了单个词语或短语，没有构成完整句子 → is_correct: false, quality: "fair"（0分）
- 无法辨认或未书写任何内容 → is_correct: false, quality: "poor"（0分）
- 看不清楚 → is_correct: null, quality: "unknown"

注意：不要求书写工整，只要能辨认且内容是完整句子即可得分。

只输出JSON：{"is_correct": true/false/null, "quality": "excellent/good/fair/poor/unknown", "detail": "简短描述，包括识别出的文字内容"}""",
    },
}


def _messages_to_ark_input(messages: list) -> list:
    """
    将 OpenAI chat messages 格式转换为 Volcengine Responses API 的 input 格式。
    system role 合并进第一条 user 消息的文本前缀。
    """
    system_text = ""
    ark_input = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            system_text = content if isinstance(content, str) else ""
            continue
        # user / assistant
        if isinstance(content, str):
            parts = []
            if system_text:
                parts.append({"type": "input_text", "text": system_text})
                system_text = ""
            parts.append({"type": "input_text", "text": content})
            ark_input.append({"role": role, "content": parts})
        elif isinstance(content, list):
            parts = []
            if system_text:
                parts.append({"type": "input_text", "text": system_text})
                system_text = ""
            for item in content:
                t = item.get("type", "")
                if t in ("text",):
                    parts.append({"type": "input_text", "text": item["text"]})
                elif t == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    parts.append({"type": "input_image", "image_url": url})
                elif t == "video_url":
                    # Volcengine Responses API 不支持 video_url，跳过（由调用方降级处理）
                    raise ValueError("Volcengine Responses API 不支持 video_url，请使用 SiliconFlow")
            ark_input.append({"role": role, "content": parts})
    return ark_input


def _call_vlm_ark(messages: list, timeout: float = 60.0) -> str:
    """
    调用 Volcengine ARK Responses API（/api/v3/responses）
    """
    api_key = os.getenv("ARK_API_KEY")
    base_url = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    model = os.getenv("VISION_EVAL_MODEL")  # 必须是 ep-xxx 视觉接入点
    if not api_key:
        raise ValueError("未配置 ARK_API_KEY")
    if not model:
        raise ValueError("未配置 VISION_EVAL_MODEL (需要 ep-xxx 视觉接入点)")

    ark_input = _messages_to_ark_input(messages)
    payload = {
        "model": model,
        "input": ark_input,
        "max_output_tokens": 600,
        "temperature": 0.05,
    }

    client = get_shared_httpx_client()
    resp = client.post(
        f"{base_url}/responses",
        json=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )
    if resp.status_code != 200:
        print(f"[VisionEval][ARK] ⚠️ API 返回 status={resp.status_code}, body_chars={len(resp.text)}")
    resp.raise_for_status()
    data = resp.json()
    # Responses API: output[].content[].text
    # 先尝试提取完整或部分输出（incomplete_details=length 时仍可能有内容）
    for item in data.get("output", []):
        for part in item.get("content", []):
            if part.get("type") == "output_text" and part.get("text", "").strip():
                return part["text"].strip()
    # 输出被截断（reason=length）且无有效内容 → 告知上层回退
    incomplete = data.get("incomplete_details", {})
    if incomplete.get("reason") == "length":
        raise ValueError(f"ARK 输出被截断(max_output_tokens不足)，回退SiliconFlow")
    raise ValueError(f"ARK Responses API 返回格式异常: {str(data)[:300]}")


def _call_vlm_ark_chat(messages: list, timeout: float = 90.0) -> str:
    """
    调用 Volcengine ARK 标准 Chat Completions API（/api/v3/chat/completions）
    支持 video_url 格式，使用 doubao-seed 等多模态模型。
    """
    api_key = os.getenv("ARK_API_KEY")
    base_url = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    from src.llm.http_client_pool import get_active_ark_model
    model = os.getenv("ARK_VIDEO_MODEL") or os.getenv("VISION_EVAL_MODEL") or get_active_ark_model()

    if not api_key:
        raise ValueError("未配置 ARK_API_KEY")
    if not model:
        raise ValueError("未配置 ARK 视频模型，请设置 ARK_VIDEO_MODEL 或 VISION_EVAL_MODEL")

    if not model.startswith("ep-"):
        print(f"[VisionEval][ARK-Chat] ⚠️ 当前视频模型不是视觉接入点: {model}")

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 600,
        "temperature": 0.05,
    }

    client = get_shared_httpx_client()
    resp = client.post(
        f"{base_url}/chat/completions",
        json=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )
    if resp.status_code != 200:
        print(f"[VisionEval][ARK-Chat] ⚠️ API 返回 status={resp.status_code}, body_chars={len(resp.text)}")
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"].strip()
    print(f"[VisionEval][ARK-Chat] ✅ {model} 调用成功")
    return content


def _call_vlm_siliconflow(messages: list, timeout: float = 60.0) -> str:
    """
    调用 SiliconFlow VLM API（/v1/chat/completions）
    """
    api_key = os.getenv("SILICONFLOW_API_KEY")
    base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    model = os.getenv("SILICONFLOW_VISION_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")

    if not api_key:
        raise ValueError("未配置 SILICONFLOW_API_KEY")

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.05,
    }

    client = get_shared_httpx_client()
    resp = client.post(
        f"{base_url}/chat/completions",
        json=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )
    if resp.status_code != 200:
        print(f"[VisionEval][SF] ⚠️ API 返回 status={resp.status_code}, body_chars={len(resp.text)}")
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _call_vlm(messages: list, timeout: float = 60.0, force_siliconflow: bool = False) -> str:
    """
    统一 VLM 调用入口：优先 Volcengine ARK，回退 SiliconFlow。
    force_siliconflow=True 时跳过 ARK（如 video_url 不被 ARK 支持时）。
    """
    vision_model = os.getenv("VISION_EVAL_MODEL", "")
    use_ark = bool(os.getenv("ARK_API_KEY")) and vision_model.startswith("ep-") and not force_siliconflow

    if use_ark:
        try:
            result = _call_vlm_ark(messages, timeout=timeout)
            print(f"[VisionEval] ✅ ARK Responses API 调用成功")
            return result
        except ValueError as e:
            # video_url 不支持等明确错误 → 回退
            print(f"[VisionEval] ⚠️ ARK 不支持，回退 SiliconFlow: {type(e).__name__}")
        except Exception as e:
            print(f"[VisionEval] ⚠️ ARK 调用失败，回退 SiliconFlow: {type(e).__name__}")

    return _call_vlm_siliconflow(messages, timeout=timeout)


def _parse_vlm_json(content: str) -> Dict[str, Any]:
    """从 VLM 返回中提取 JSON"""
    if "```" in content:
        match = re.search(r"```(?:json)?(.*?)```", content, re.DOTALL)
        if match:
            content = match.group(1).strip()
    json_match = re.search(r"\{[\s\S]*\}", content)
    if json_match:
        return json.loads(json_match.group())
    return {"is_correct": None, "quality": "unknown", "detail": content}


def _normalize_three_step_action_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """为三步动作补齐稳定的 steps_completed 字段。"""
    if not isinstance(result, dict):
        return result

    steps_completed = result.get("steps_completed")
    if isinstance(steps_completed, str) and steps_completed.isdigit():
        steps_completed = int(steps_completed)

    if not isinstance(steps_completed, int):
        detail = str(result.get("detail", ""))
        steps_completed = 0
        direct_count_match = re.search(r"([0-3])\s*步", detail)
        if direct_count_match:
            steps_completed = int(direct_count_match.group(1))
        else:
            keyword_groups = [
                (["举手", "举起", "抬手"], ["没举", "未举", "没有举"]),
                (["握拳", "拳头", "握成拳"], ["没握拳", "未握拳", "没有握拳"]),
                (["放到胸前", "放在胸前", "胸前", "靠近胸口"], ["没放到胸前", "未放到胸前", "没有放到胸前"]),
            ]
            for positive_keywords, negative_keywords in keyword_groups:
                if any(keyword in detail for keyword in negative_keywords):
                    continue
                if any(keyword in detail for keyword in positive_keywords):
                    steps_completed += 1

        if steps_completed == 0 and result.get("is_correct") is True:
            steps_completed = 3
        elif steps_completed == 0 and result.get("quality") == "fair":
            steps_completed = 1

    result["steps_completed"] = max(0, min(3, int(steps_completed)))
    return result


def _build_result(result: Dict, task_id: str, elapsed_ms: float, source: str) -> Dict[str, Any]:
    """标准化输出格式"""
    if task_id == "language_3step_action":
        result = _normalize_three_step_action_result(result)

    is_correct = result.get("is_correct")
    quality = result.get("quality", result.get("quality_level", "unknown"))
    detail = result.get("detail", "")
    quality_to_cognitive = {
        "excellent": "正常", "good": "正常",
        "fair": "轻度异常", "poor": "异常", "unknown": "无法判断",
    }
    return {
        "success": True,
        "is_correct": is_correct,
        "quality_level": quality,
        "cognitive_performance": quality_to_cognitive.get(quality, "无法判断"),
        "is_complete": True,
        "evaluation_detail": f"{source}评估: {detail}",
        "need_followup": False,
        "confidence": 0.85 if is_correct is not None else 0.0,
        "source": source,
        "steps_completed": result.get("steps_completed"),
        "elapsed_ms": elapsed_ms,
    }


def _split_base64_payload(media_base64: str) -> tuple[str, str]:
    raw = media_base64 or ""
    detected_mime = ""
    if raw.startswith("data:") and "," in raw:
        header, raw = raw.split(",", 1)
        match = re.match(r"data:([^;]+);base64", header)
        if match:
            detected_mime = match.group(1)
    elif "," in raw:
        raw = raw.split(",", 1)[1]
    return raw, detected_mime


def _mime_to_suffix(mime_type: str) -> str:
    mapping = {
        "video/webm": ".webm",
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/x-matroska": ".mkv",
        "video/avi": ".avi",
    }
    return mapping.get((mime_type or "").lower(), ".video")


def _prepare_video_for_vlm(video_base64: str, mime_type: str) -> tuple[str, str]:
    raw_b64, detected_mime = _split_base64_payload(video_base64)
    effective_mime = detected_mime or mime_type or "video/webm"

    if effective_mime.lower() == "video/mp4":
        return raw_b64, "video/mp4"

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        print(f"[VisionEval] ⚠️ ffmpeg 不可用，继续使用原始视频格式: {effective_mime}")
        return raw_b64, effective_mime

    src_path = ""
    dst_path = ""
    try:
        video_bytes = base64.b64decode(raw_b64, validate=True)
    except Exception:
        video_bytes = base64.b64decode(raw_b64)

    try:
        with tempfile.NamedTemporaryFile(suffix=_mime_to_suffix(effective_mime), delete=False) as src_file:
            src_file.write(video_bytes)
            src_path = src_file.name
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as dst_file:
            dst_path = dst_file.name

        cmd = [
            ffmpeg_path,
            "-y",
            "-i",
            src_path,
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-movflags",
            "+faststart",
            "-crf",
            "28",
            dst_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            stderr = (proc.stderr or proc.stdout or "")[-500:]
            raise RuntimeError(stderr.strip() or "ffmpeg 转码失败")

        with open(dst_path, "rb") as f:
            mp4_bytes = f.read()
        if not mp4_bytes:
            raise RuntimeError("ffmpeg 未生成有效 mp4 文件")

        print(
            f"[VisionEval] 🎬 视频转码完成: {effective_mime} -> video/mp4, "
            f"{len(video_bytes)}B -> {len(mp4_bytes)}B"
        )
        return base64.b64encode(mp4_bytes).decode("utf-8"), "video/mp4"
    except Exception as e:
        print(f"[VisionEval] ⚠️ 视频转码失败，继续使用原始格式: {type(e).__name__}")
        return raw_b64, effective_mime
    finally:
        for path in (src_path, dst_path):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass


def evaluate_video_with_vlm(
    video_base64: str,
    task_id: str,
    mime_type: str = "video/webm",
    extra_context: str = "",
) -> Dict[str, Any]:
    """
    使用 SiliconFlow video_url 格式评估视频（最准确）

    Args:
        video_base64: base64 编码的视频（可含 data:video/... 前缀）
        task_id: MMSE 任务 ID
        mime_type: 视频 MIME 类型（浏览器录的一般是 video/webm）
        extra_context: 额外上下文信息
    """
    start_time = time.time()

    task_config = VISION_TASK_PROMPTS.get(task_id)
    if not task_config:
        return {"success": False, "error": f"不支持的视觉任务: {task_id}",
                "is_correct": None, "quality_level": "unknown"}

    raw_b64, effective_mime = _prepare_video_for_vlm(video_base64, mime_type)

    user_prompt = task_config["prompt"]
    if extra_context:
        user_prompt += f"\n\n补充信息：{extra_context}"

    print(f"[VisionEval] 🎬 视频直评载荷: input_mime={mime_type}, send_mime={effective_mime}")
    data_uri = f"data:{effective_mime};base64,{raw_b64}"
    messages = [
        {"role": "system", "content": task_config["system"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {
                        "url": data_uri,
                        "detail": "high",
                        "max_frames": 16,
                        "fps": 2,
                    },
                },
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    # 🔥 优先用 ARK chat/completions（doubao-seed 支持 video_url），降级到 SiliconFlow
    source = "视频"
    try:
        if os.getenv("ARK_API_KEY"):
            content = _call_vlm_ark_chat(messages, timeout=90.0)
            source = "视频(ARK)"
        else:
            content = _call_vlm_siliconflow(messages, timeout=90.0)
            source = "视频(SF)"
    except Exception as e:
        print(f"[VisionEval] ⚠️ 视频模式主路径失败: {type(e).__name__}")
        try:
            content = _call_vlm_siliconflow(messages, timeout=90.0)
            source = "视频(SF降级)"
        except Exception as e2:
            raise RuntimeError(
                f"视频评估全部失败: ARK={type(e).__name__}, SF={type(e2).__name__}"
            ) from e2

    print(f"[VisionEval] 🎬 VLM 视频返回: text_chars={len(content)}")
    result = _parse_vlm_json(content)
    elapsed = (time.time() - start_time) * 1000
    print(f"[VisionEval] ✅ 视频评估完成: task={task_id}, result_fields={len(result)} ({elapsed:.0f}ms)")
    return _build_result(result, task_id, elapsed, source)


def evaluate_frames_with_vlm(
    frames_base64: list,
    task_id: str,
    extra_context: str = "",
) -> Dict[str, Any]:
    """
    多帧评估（降级方案）：将多张摄像头截图作为时序图片序列发送给 VL 模型
    """
    start_time = time.time()

    task_config = VISION_TASK_PROMPTS.get(task_id)
    if not task_config:
        return {"success": False, "error": f"不支持的视觉任务: {task_id}",
                "is_correct": None, "quality_level": "unknown"}

    if not frames_base64 or len(frames_base64) == 0:
        return {"success": False, "error": "未收到任何帧数据",
                "is_correct": None, "quality_level": "unknown"}

    n_frames = len(frames_base64)
    print(f"[VisionEval] 🎞️ 多帧降级: {n_frames} 帧, task={task_id}")

    user_prompt = task_config["prompt"]
    if extra_context:
        user_prompt += f"\n\n补充信息：{extra_context}"

    content_parts = []
    for i, frame_b64 in enumerate(frames_base64):
        raw = frame_b64
        if "," in raw:
            raw = raw.split(",", 1)[1]
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{raw}",
                "detail": "low",
            },
        })

    time_hint = f"以上是按时间顺序拍摄的 {n_frames} 张连续截图，请综合所有图片判断患者的动作变化。\n\n"
    content_parts.append({"type": "text", "text": time_hint + user_prompt})

    messages = [
        {"role": "system", "content": task_config["system"]},
        {"role": "user", "content": content_parts},
    ]

    content = _call_vlm(messages, timeout=60.0)
    print(f"[VisionEval] 🎞️ VLM 多帧返回: text_chars={len(content)}")
    result = _parse_vlm_json(content)
    elapsed = (time.time() - start_time) * 1000
    print(f"[VisionEval] ✅ 多帧评估完成: task={task_id}, {n_frames}帧, result_fields={len(result)} ({elapsed:.0f}ms)")
    return _build_result(result, task_id, elapsed, f"多帧({n_frames}帧)")


def evaluate_hybrid(
    task_id: str,
    video_base64: str = "",
    mime_type: str = "video/webm",
    frames_base64: list = None,
    extra_context: str = "",
) -> Dict[str, Any]:
    """
    混合评估：优先用 video_url（最准确），失败则自动降级到多帧

    Args:
        task_id: MMSE 任务 ID
        video_base64: 视频 base64（可选）
        mime_type: 视频 MIME 类型
        frames_base64: 多帧截图列表（可选，作为降级）
        extra_context: 额外上下文
    """
    mediapipe_result = None
    mediapipe_error = None

    def get_mediapipe_result() -> Optional[Dict[str, Any]]:
        nonlocal mediapipe_result, mediapipe_error
        if mediapipe_result is not None or mediapipe_error is not None:
            return mediapipe_result
        if task_id not in {"language_reading_close_eyes", "language_3step_action"}:
            return None
        if not video_base64 and not frames_base64:
            return None
        try:
            from src.tools.agent_tools.mediapipe_vision import evaluate_with_mediapipe
            mediapipe_result = evaluate_with_mediapipe(
                task_id=task_id,
                video_base64=video_base64,
                frames_base64=frames_base64,
            )
        except Exception as e:
            mediapipe_error = type(e).__name__
            print(f"[VisionEval] ⚠️ MediaPipe 兜底失败: {mediapipe_error}")
        return mediapipe_result

    # 策略1: 优先视频
    video_error = None
    if video_base64:
        try:
            print(f"[VisionEval] 🎬 尝试视频模式 (video_url)...")
            video_result = evaluate_video_with_vlm(video_base64, task_id, mime_type, extra_context)
            if video_result.get("is_correct") is not None:
                return video_result
            video_error = "video_result_unknown"
            print(f"[VisionEval] ⚠️ 视频模式结果不确定，准备回退")
        except Exception as e:
            video_error = type(e).__name__
            print(f"[VisionEval] ⚠️ 视频模式失败，降级到多帧: {video_error}")

    # 策略2: 多帧降级
    frame_error = None
    if frames_base64 and len(frames_base64) > 0:
        try:
            result = evaluate_frames_with_vlm(frames_base64, task_id, extra_context)
            if video_error:
                result["video_fallback_reason"] = video_error
            if result.get("is_correct") is not None:
                return result
            frame_error = "frames_result_unknown"
            print(f"[VisionEval] ⚠️ 多帧模式结果不确定，准备回退 MediaPipe")
        except Exception as e:
            frame_error = type(e).__name__
            print(f"[VisionEval] ❌ 多帧模式也失败: {frame_error}")

    mediapipe_result = get_mediapipe_result()
    if mediapipe_result and mediapipe_result.get("success"):
        if video_error:
            mediapipe_result["video_fallback_reason"] = video_error
        if frame_error:
            mediapipe_result["frame_fallback_reason"] = frame_error
        print(f"[VisionEval] ✅ 使用 MediaPipe 结果兜底: task={task_id}, result_fields={len(mediapipe_result)}")
        return mediapipe_result

    if frame_error:
        return {
            "success": False, "error": frame_error,
            "is_correct": None, "quality_level": "unknown",
            "cognitive_performance": "无法判断", "is_complete": False,
            "evaluation_detail": f"视频和多帧评估均失败: {frame_error}",
            "confidence": 0.0, "source": "hybrid",
        }

    if video_error:
        return {
            "success": False, "error": video_error,
            "is_correct": None, "quality_level": "unknown",
            "cognitive_performance": "无法判断", "is_complete": False,
            "evaluation_detail": f"视频评估失败且无可用兜底: {video_error}",
            "confidence": 0.0, "source": "hybrid",
        }

    return {"success": False, "error": "未收到视频或帧数据",
            "is_correct": None, "quality_level": "unknown"}


def evaluate_image_with_vlm(
    image_base64: str,
    task_id: str,
    extra_context: str = "",
) -> Dict[str, Any]:
    """
    调用 SiliconFlow 视觉大模型评估图片（降级模式）
    """
    start_time = time.time()

    task_config = VISION_TASK_PROMPTS.get(task_id)
    if not task_config:
        return {"success": False, "error": f"不支持的视觉任务: {task_id}",
                "is_correct": None, "quality_level": "unknown"}

    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    user_prompt = task_config["prompt"].replace("这段视频", "这张图片").replace("视频中", "图片中")
    if extra_context:
        user_prompt += f"\n\n补充信息：{extra_context}"

    messages = [
        {"role": "system", "content": task_config["system"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    try:
        content = _call_vlm(messages, timeout=30.0)
        print(f"[VisionEval] 🔍 VLM 图片返回: text_chars={len(content)}")
        result = _parse_vlm_json(content)
        elapsed = (time.time() - start_time) * 1000
        print(f"[VisionEval] ✅ 图片评估完成: task={task_id}, result_fields={len(result)} ({elapsed:.0f}ms)")
        return _build_result(result, task_id, elapsed, "图片")

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        error_type = type(e).__name__
        print(f"[VisionEval] ❌ 图片评估失败 ({elapsed:.0f}ms): {error_type}")
        return {
            "success": False, "error": error_type,
            "is_correct": None, "quality_level": "unknown",
            "cognitive_performance": "无法判断", "is_complete": False,
            "evaluation_detail": f"图片评估失败: {error_type}",
            "confidence": 0.0, "source": "vision_model",
        }
