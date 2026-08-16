"""
MediaPipe 本地视觉检测 — 替代 SiliconFlow VLM API

支持：
1. 闭眼检测 (EAR — Eye Aspect Ratio)
2. 三步动作兜底检测 (上半身启发式：举手/握拳/放到胸前)

全 CPU，毫秒级，零 API 调用。
"""

import base64
import io
import math
import time
import tempfile
import os
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

import mediapipe as mp

# ───────── 全局初始化（懒加载）─────────
_face_mesh = None
_hands = None
_pose = None


def _get_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,  # 开启虹膜 landmark
            min_detection_confidence=0.5,
        )
    return _face_mesh


def _get_hands():
    global _hands
    if _hands is None:
        _hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _hands


def _get_pose():
    global _pose
    if _pose is None:
        _pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        )
    return _pose


# ───────── 工具函数 ─────────

def _b64_to_cv2(b64_str: str) -> Optional[np.ndarray]:
    """base64 → OpenCV BGR 图像"""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(b64_str)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[MediaPipe] ⚠️ base64 解码失败: {type(e).__name__}")
        return None


def _video_b64_to_frames(video_b64: str, max_frames: int = 16, fps: float = 2.0) -> List[np.ndarray]:
    """base64 视频 → 帧列表（抽帧）"""
    if "," in video_b64:
        video_b64 = video_b64.split(",", 1)[1]
    try:
        video_bytes = base64.b64decode(video_b64)
        # 写临时文件让 OpenCV 读取
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(video_bytes)
            tmp_path = f.name

        cap = cv2.VideoCapture(tmp_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 计算抽帧间隔
        if total_frames <= 0:
            os.unlink(tmp_path)
            return []
        interval = max(1, int(video_fps / fps))
        frames = []
        idx = 0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                frames.append(frame)
            idx += 1
        cap.release()
        os.unlink(tmp_path)
        return frames
    except Exception as e:
        print(f"[MediaPipe] ⚠️ 视频解码失败: {type(e).__name__}")
        return []


def _dist(p1, p2) -> float:
    """两点欧氏距离"""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


# ───────── 闭眼检测 (EAR) ─────────

# MediaPipe Face Mesh 眼睛关键点索引
# 右眼: p1=33, p2=160, p3=158, p4=133, p5=153, p6=144
# 左眼: p1=362, p2=385, p3=387, p4=263, p5=373, p6=380
_RIGHT_EYE = [33, 160, 158, 133, 153, 144]
_LEFT_EYE = [362, 385, 387, 263, 373, 380]

# EAR 阈值：低于此值视为闭眼
_EAR_THRESHOLD = 0.21


def _compute_ear(landmarks, eye_indices) -> float:
    """计算单只眼睛的 Eye Aspect Ratio"""
    p = [landmarks[i] for i in eye_indices]
    # EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    vertical_1 = _dist(p[1], p[5])
    vertical_2 = _dist(p[2], p[4])
    horizontal = _dist(p[0], p[3])
    if horizontal < 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def detect_eye_close(frame: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    单帧闭眼检测。
    返回 {"closed": bool, "ear": float} 或 None（未检测到人脸）
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _get_face_mesh().process(rgb)
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    ear_right = _compute_ear(landmarks, _RIGHT_EYE)
    ear_left = _compute_ear(landmarks, _LEFT_EYE)
    avg_ear = (ear_right + ear_left) / 2.0

    return {"closed": avg_ear < _EAR_THRESHOLD, "ear": round(avg_ear, 3)}


def evaluate_eye_close_multiframe(frames: List[np.ndarray]) -> Dict[str, Any]:
    """
    多帧闭眼检测。判断是否有「从睁眼到闭眼」的动作变化。
    """
    if not frames:
        return {"is_correct": None, "quality": "unknown", "detail": "无帧数据"}

    results = []
    for f in frames:
        r = detect_eye_close(f)
        if r is not None:
            results.append(r)

    if not results:
        return {"is_correct": None, "quality": "unknown", "detail": "未检测到人脸"}

    closed_count = sum(1 for r in results if r["closed"])
    total = len(results)
    closed_ratio = closed_count / total
    avg_ear = sum(r["ear"] for r in results) / total

    # 判断逻辑：
    # - 有明确闭眼帧（>=20%）且有睁眼帧 → 有闭眼动作
    # - 大部分帧闭眼（>=60%） → 也算正确
    has_open = any(not r["closed"] for r in results)
    has_close = any(r["closed"] for r in results)

    if has_close and has_open and closed_ratio >= 0.15:
        quality = "excellent" if closed_ratio >= 0.3 else "good"
        return {
            "is_correct": True, "quality": quality,
            "detail": f"检测到闭眼动作 (闭眼帧{closed_count}/{total}, EAR={avg_ear:.3f})",
        }
    elif closed_ratio >= 0.6:
        return {
            "is_correct": True, "quality": "good",
            "detail": f"大部分时间闭眼 (闭眼帧{closed_count}/{total}, EAR={avg_ear:.3f})",
        }
    elif has_close:
        # 有少量闭眼帧但不够明显，可能是眨眼
        return {
            "is_correct": False, "quality": "poor",
            "detail": f"可能只是眨眼 (闭眼帧{closed_count}/{total}, EAR={avg_ear:.3f})",
        }
    else:
        return {
            "is_correct": False, "quality": "poor",
            "detail": f"眼睛始终睁开 (闭眼帧{closed_count}/{total}, EAR={avg_ear:.3f})",
        }


# ───────── 手势"5"检测（手指计数）─────────

# MediaPipe Hands landmark 索引
_FINGER_TIPS = [4, 8, 12, 16, 20]        # 拇指尖、食指尖、中指尖、无名指尖、小指尖
_FINGER_PIPS = [3, 6, 10, 14, 18]        # PIP 关节
_FINGER_MCPS = [2, 5, 9, 13, 17]         # MCP 关节
_WRIST = 0


def _count_fingers(hand_landmarks, handedness: str = "Right") -> int:
    """
    计算伸出的手指数量。
    handedness: "Right" 或 "Left"（MediaPipe 返回的是镜像标签）
    """
    lm = hand_landmarks.landmark
    count = 0

    # 拇指：比较 tip.x 和 IP(index 3).x 的关系
    # 注意 MediaPipe 的 handedness 是镜像的
    if handedness == "Right":
        if lm[4].x < lm[3].x:  # 右手拇指向左伸出
            count += 1
    else:
        if lm[4].x > lm[3].x:  # 左手拇指向右伸出
            count += 1

    # 其余4根手指：tip.y < PIP.y 表示伸直（y 轴朝下）
    for tip_idx, pip_idx in zip(_FINGER_TIPS[1:], _FINGER_PIPS[1:]):
        if lm[tip_idx].y < lm[pip_idx].y:
            count += 1

    return count


def _hand_near_chest(hand_landmarks, face_landmarks=None) -> bool:
    """
    判断手是否在胸口附近。
    简单启发式：手的 wrist.y 在画面中部偏下（0.3~0.8），
    如果有人脸 landmark，则手应在下巴以下。
    """
    wrist_y = hand_landmarks.landmark[_WRIST].y
    wrist_x = hand_landmarks.landmark[_WRIST].x

    # 基本范围检查：手在画面中部区域
    if wrist_y < 0.25 or wrist_y > 0.85:
        return False
    if wrist_x < 0.15 or wrist_x > 0.85:
        return False

    # 如果有人脸信息，检查手是否在下巴以下
    if face_landmarks:
        chin_y = face_landmarks.landmark[152].y  # 下巴
        if wrist_y < chin_y - 0.05:
            return False  # 手在脸部以上，不太像胸口

    return True


def detect_hand_gesture_5(frame: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    单帧手势"5"检测。
    返回 {"is_five": bool, "fingers": int, "near_chest": bool} 或 None
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = _get_hands().process(rgb)

    if not hand_results.multi_hand_landmarks:
        return None

    # 同时检测人脸用于胸口判断
    face_results = _get_face_mesh().process(rgb)
    face_lm = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None

    best = None
    for i, hand_lm in enumerate(hand_results.multi_hand_landmarks):
        handedness = "Right"
        if hand_results.multi_handedness and i < len(hand_results.multi_handedness):
            handedness = hand_results.multi_handedness[i].classification[0].label

        fingers = _count_fingers(hand_lm, handedness)
        near_chest = _hand_near_chest(hand_lm, face_lm)

        if best is None or fingers > best["fingers"]:
            best = {
                "is_five": fingers == 5,
                "fingers": fingers,
                "near_chest": near_chest,
            }

    return best


def evaluate_hand_gesture_5_multiframe(frames: List[np.ndarray]) -> Dict[str, Any]:
    """
    多帧手势"5"检测。判断是否在胸口摆出5的手势。
    """
    if not frames:
        return {"is_correct": None, "quality": "unknown", "detail": "无帧数据"}

    results = []
    for f in frames:
        r = detect_hand_gesture_5(f)
        if r is not None:
            results.append(r)

    if not results:
        return {"is_correct": None, "quality": "unknown", "detail": "未检测到手部"}

    five_count = sum(1 for r in results if r["is_five"])
    chest_five_count = sum(1 for r in results if r["is_five"] and r["near_chest"])
    total = len(results)
    max_fingers = max(r["fingers"] for r in results)
    avg_fingers = sum(r["fingers"] for r in results) / total

    if chest_five_count >= 1:
        return {
            "is_correct": True, "quality": "excellent",
            "detail": f"胸口摆出5手势 ({chest_five_count}/{total}帧, 平均{avg_fingers:.1f}指)",
        }
    elif five_count >= 1:
        return {
            "is_correct": True, "quality": "good",
            "detail": f"摆出5手势但位置不在胸口 ({five_count}/{total}帧, 平均{avg_fingers:.1f}指)",
        }
    elif max_fingers >= 4:
        return {
            "is_correct": True, "quality": "good",
            "detail": f"手势接近5 (最多{max_fingers}指, 平均{avg_fingers:.1f}指)",
        }
    elif max_fingers >= 1:
        return {
            "is_correct": False, "quality": "poor",
            "detail": f"手势不正确 (最多{max_fingers}指, 平均{avg_fingers:.1f}指)",
        }
    else:
        return {
            "is_correct": False, "quality": "poor",
            "detail": f"未做出手势 (平均{avg_fingers:.1f}指)",
        }


def _is_visible(landmark, threshold: float = 0.35) -> bool:
    return getattr(landmark, "visibility", 1.0) >= threshold


def detect_three_step_action_pose(frame: np.ndarray) -> Optional[Dict[str, Any]]:
    """检测适合胸部以上画面的三步动作：举手、握拳、放到胸前。"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = _get_hands().process(rgb)
    pose_results = _get_pose().process(rgb)

    if not hand_results.multi_hand_landmarks:
        return None

    shoulder_y = 0.48
    chest_y = 0.58
    torso_left_x = 0.25
    torso_right_x = 0.75
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark
        left_shoulder = lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        visible_shoulders = [p for p in [left_shoulder, right_shoulder] if _is_visible(p, 0.2)]
        if visible_shoulders:
            shoulder_y = sum(p.y for p in visible_shoulders) / len(visible_shoulders)
        if _is_visible(left_shoulder, 0.2) and _is_visible(right_shoulder, 0.2):
            torso_left_x = min(left_shoulder.x, right_shoulder.x) - 0.12
            torso_right_x = max(left_shoulder.x, right_shoulder.x) + 0.12
            chest_y = shoulder_y + 0.16

    raised = False
    fist = False
    hand_to_chest = False
    best_fingers = None

    for i, hand_lm in enumerate(hand_results.multi_hand_landmarks):
        handedness = "Right"
        if hand_results.multi_handedness and i < len(hand_results.multi_handedness):
            handedness = hand_results.multi_handedness[i].classification[0].label

        fingers = _count_fingers(hand_lm, handedness)
        wrist_y = hand_lm.landmark[_WRIST].y
        wrist_x = hand_lm.landmark[_WRIST].x
        middle_mcp_y = hand_lm.landmark[9].y
        middle_mcp_x = hand_lm.landmark[9].x

        # y 越小越靠上；举手通常会接近或高于肩线。
        raised = raised or wrist_y <= shoulder_y + 0.12 or middle_mcp_y <= shoulder_y + 0.08
        fist = fist or fingers <= 1
        in_torso_x = (
            torso_left_x <= wrist_x <= torso_right_x
            or torso_left_x <= middle_mcp_x <= torso_right_x
        )
        near_chest_y = abs(wrist_y - chest_y) <= 0.22 or abs(middle_mcp_y - chest_y) <= 0.20
        below_shoulder = wrist_y >= shoulder_y - 0.02 or middle_mcp_y >= shoulder_y - 0.04
        hand_to_chest = hand_to_chest or (in_torso_x and near_chest_y and below_shoulder)
        best_fingers = fingers if best_fingers is None else max(best_fingers, fingers)

    return {
        "raise_hand": raised,
        "fist": fist,
        "hand_to_chest": hand_to_chest,
        "max_fingers": best_fingers if best_fingers is not None else 0,
    }


def evaluate_three_step_action_multiframe(frames: List[np.ndarray]) -> Dict[str, Any]:
    if not frames:
        return {"is_correct": None, "quality": "unknown", "steps_completed": 0, "detail": "无帧数据"}

    results = []
    for frame in frames:
        result = detect_three_step_action_pose(frame)
        if result is not None:
            results.append(result)

    if not results:
        return {"is_correct": None, "quality": "unknown", "steps_completed": 0, "detail": "未检测到可用人体姿态"}

    raise_indices = [i for i, item in enumerate(results) if item["raise_hand"]]
    fist_indices = [i for i, item in enumerate(results) if item["fist"]]
    chest_indices = [i for i, item in enumerate(results) if item["hand_to_chest"]]

    raise_idx = raise_indices[0] if raise_indices else None
    fist_idx = None
    chest_idx = None

    if raise_idx is not None:
        fist_after_raise = [i for i in fist_indices if i >= raise_idx]
        if fist_after_raise:
            fist_idx = fist_after_raise[0]

    if fist_idx is not None:
        chest_after_fist = [i for i in chest_indices if i >= fist_idx]
        if chest_after_fist:
            chest_idx = chest_after_fist[0]
    elif raise_idx is not None:
        chest_after_raise = [i for i in chest_indices if i > raise_idx]
        if chest_after_raise:
            chest_idx = chest_after_raise[0]

    steps_completed = 0
    observed = []
    if raise_idx is not None:
        steps_completed += 1
        observed.append("举起手")
    if fist_idx is not None:
        steps_completed += 1
        observed.append("握拳")
    if chest_idx is not None:
        steps_completed += 1
        observed.append("手放到胸前")

    if steps_completed >= 3:
        quality = "excellent"
        is_correct = True
    elif steps_completed == 2:
        quality = "fair"
        is_correct = False
    elif steps_completed <= 1:
        quality = "poor"
        is_correct = False
    else:
        quality = "unknown"
        is_correct = None

    max_fingers = max(item.get("max_fingers", 0) for item in results)
    detail = "、".join(observed) if observed else "未观察到明确的上肢三步动作"
    detail = f"{detail} ({steps_completed}/3, 有效手部帧{len(results)}/{len(frames)}, 最大伸指{max_fingers})"
    return {
        "is_correct": is_correct,
        "quality": quality,
        "steps_completed": steps_completed,
        "detail": detail,
    }


# ───────── 统一入口 ─────────

def evaluate_with_mediapipe(
    task_id: str,
    video_base64: str = "",
    frames_base64: List[str] = None,
    image_base64: str = "",
) -> Dict[str, Any]:
    """
    MediaPipe 统一评估入口。返回格式与 VLM 版本一致。

    Args:
        task_id: "language_reading_close_eyes" 或 "language_3step_action"
        video_base64: base64 视频
        frames_base64: base64 帧列表
        image_base64: 单张 base64 图片
    """
    start = time.time()

    # 1. 收集帧
    frames: List[np.ndarray] = []

    if video_base64:
        frames = _video_b64_to_frames(video_base64)
        print(f"[MediaPipe] 🎬 视频解码: {len(frames)} 帧")

    if not frames and frames_base64:
        for fb in frames_base64:
            img = _b64_to_cv2(fb)
            if img is not None:
                frames.append(img)
        print(f"[MediaPipe] 🎞️ 多帧解码: {len(frames)} 帧")

    if not frames and image_base64:
        img = _b64_to_cv2(image_base64)
        if img is not None:
            frames = [img]
        print(f"[MediaPipe] 🖼️ 单帧解码: {len(frames)} 帧")

    if not frames:
        return {
            "success": False, "error": "无法解码任何帧数据",
            "is_correct": None, "quality_level": "unknown",
            "cognitive_performance": "无法判断", "is_complete": False,
            "evaluation_detail": "MediaPipe: 无帧数据",
            "confidence": 0.0, "source": "mediapipe",
        }

    # 2. 按任务分发
    if task_id == "language_reading_close_eyes":
        result = evaluate_eye_close_multiframe(frames)
    elif task_id == "language_3step_action":
        result = evaluate_three_step_action_multiframe(frames)
    else:
        return {
            "success": False, "error": f"MediaPipe 不支持的任务: {task_id}",
            "is_correct": None, "quality_level": "unknown",
        }

    elapsed = (time.time() - start) * 1000

    # 3. 标准化输出
    is_correct = result.get("is_correct")
    quality = result.get("quality", "unknown")
    detail = result.get("detail", "")
    quality_to_cognitive = {
        "excellent": "正常", "good": "正常",
        "fair": "轻度异常", "poor": "异常", "unknown": "无法判断",
    }

    print(f"[MediaPipe] ✅ {task_id}: correct={is_correct}, quality={quality}, {elapsed:.0f}ms, detail_chars={len(str(detail))}")

    return {
        "success": True,
        "is_correct": is_correct,
        "quality_level": quality,
        "cognitive_performance": quality_to_cognitive.get(quality, "无法判断"),
        "is_complete": True,
        "evaluation_detail": f"MediaPipe评估: {detail}",
        "need_followup": False,
        "confidence": 0.85 if is_correct is not None else 0.0,
        "source": "mediapipe",
        "steps_completed": result.get("steps_completed"),
        "elapsed_ms": round(elapsed, 1),
    }
