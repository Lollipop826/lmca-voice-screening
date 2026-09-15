"""验证两个 TTS 提速手段是否可行。

基线已测清：ping 467ms + StartSession 467ms + 合成 670ms ≈ 1.6s，
其中两个 467ms 是 CDN edge 层的固定单程开销（server-timing: edge;dur=645，
origin 仅 74~85ms），网络 RTT 只有 4.4ms。

本脚本验证：
  1. 去掉合成前的 ping 探活 —— 省 467ms
  2. StartSession 后不等 SessionStarted，直接流水线发 TaskRequest —— 若服务端
     接受，再省 467ms

每种方案连测 3 次取中位数。
"""
import asyncio
import json
import os
import statistics
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import websockets
from dotenv import load_dotenv

from src.tools.voice.ark_tts import (
    _WSS_URL,
    _EVT_START_CONN, _EVT_CONN_STARTED,
    _EVT_START_SESSION, _EVT_SESSION_STARTED, _EVT_SESSION_FINISHED,
    _EVT_SESSION_FAILED, _EVT_TASK_REQUEST, _EVT_FINISH_SESSION,
    _frame_no_id, _frame_with_sid, _parse_server_frame,
)

load_dotenv()

APP_ID = os.getenv("VOLC_APP_ID", "")
TOKEN = os.getenv("VOLC_ACCESS_TOKEN", "")
VOICE = os.getenv("VOLC_TTS_VOICE", "zh_female_vv_uranus_bigtts")
RESOURCE = os.getenv("VOLC_TTS_RESOURCE", "seed-tts-2.0")

TEXTS = ["您好，今天感觉怎么样？", "我们先做个简单的小测试。", "别着急，慢慢来就好。"]

_AUDIO_PARAMS = {
    "format": "pcm", "sample_rate": 24000,
    "speech_rate": 0, "emotion": "neutral", "emotion_scale": 3,
}


async def open_conn():
    headers = {
        "X-Api-App-Key": APP_ID, "X-Api-Access-Key": TOKEN,
        "X-Api-Resource-Id": RESOURCE, "X-Api-Connect-Id": str(uuid.uuid4()),
    }
    ws = await websockets.connect(_WSS_URL, additional_headers=headers, open_timeout=10)
    await ws.send(_frame_no_id(_EVT_START_CONN, b"{}"))
    resp = _parse_server_frame(await asyncio.wait_for(ws.recv(), timeout=10))
    if resp["event"] != _EVT_CONN_STARTED:
        await ws.close()
        raise RuntimeError(f"StartConnection 失败 event={resp['event']}")
    return ws


def _session_meta():
    return json.dumps({
        "user": {"uid": "probe"},
        "req_params": {"speaker": VOICE, "audio_params": _AUDIO_PARAMS},
    }, ensure_ascii=False).encode()


async def _drain(ws, t0, sid_expected=None):
    """读到 SessionFinished，返回首音频延迟(ms)。"""
    first = None
    while True:
        resp = _parse_server_frame(await asyncio.wait_for(ws.recv(), timeout=30))
        if resp["audio"]:
            if first is None:
                first = (time.time() - t0) * 1000
        elif resp["event"] == _EVT_SESSION_FINISHED:
            return first
        elif resp["event"] == _EVT_SESSION_FAILED:
            raise RuntimeError(f"会话失败 payload={resp.get('payload')}")


async def variant_current(ws, text):
    """现状：ping 探活 → StartSession → 等 SessionStarted → TaskRequest。"""
    t0 = time.time()
    waiter = await ws.ping()
    if waiter is not None:
        await asyncio.wait_for(waiter, timeout=5)
    sid = str(uuid.uuid4())
    await ws.send(_frame_with_sid(_EVT_START_SESSION, sid, _session_meta()))
    resp = _parse_server_frame(await asyncio.wait_for(ws.recv(), timeout=10))
    if resp["event"] != _EVT_SESSION_STARTED:
        raise RuntimeError(f"会话启动失败 event={resp['event']}")
    await ws.send(_frame_with_sid(
        _EVT_TASK_REQUEST, sid,
        json.dumps({"req_params": {"text": text}}, ensure_ascii=False).encode()))
    await ws.send(_frame_with_sid(_EVT_FINISH_SESSION, sid, b"{}"))
    return await _drain(ws, t0)


async def variant_no_ping(ws, text):
    """方案1：去掉 ping，其余不变。"""
    t0 = time.time()
    sid = str(uuid.uuid4())
    await ws.send(_frame_with_sid(_EVT_START_SESSION, sid, _session_meta()))
    resp = _parse_server_frame(await asyncio.wait_for(ws.recv(), timeout=10))
    if resp["event"] != _EVT_SESSION_STARTED:
        raise RuntimeError(f"会话启动失败 event={resp['event']}")
    await ws.send(_frame_with_sid(
        _EVT_TASK_REQUEST, sid,
        json.dumps({"req_params": {"text": text}}, ensure_ascii=False).encode()))
    await ws.send(_frame_with_sid(_EVT_FINISH_SESSION, sid, b"{}"))
    return await _drain(ws, t0)


async def variant_pipelined(ws, text):
    """方案2：去 ping + 三帧一次性发出，不等 SessionStarted。"""
    t0 = time.time()
    sid = str(uuid.uuid4())
    await ws.send(_frame_with_sid(_EVT_START_SESSION, sid, _session_meta()))
    await ws.send(_frame_with_sid(
        _EVT_TASK_REQUEST, sid,
        json.dumps({"req_params": {"text": text}}, ensure_ascii=False).encode()))
    await ws.send(_frame_with_sid(_EVT_FINISH_SESSION, sid, b"{}"))
    # SessionStarted 会先到，跳过它继续读音频
    first = None
    while True:
        resp = _parse_server_frame(await asyncio.wait_for(ws.recv(), timeout=30))
        if resp["audio"]:
            if first is None:
                first = (time.time() - t0) * 1000
        elif resp["event"] == _EVT_SESSION_FINISHED:
            return first
        elif resp["event"] == _EVT_SESSION_FAILED:
            raise RuntimeError(f"会话失败 payload={resp.get('payload')}")


VARIANTS = [
    ("现状（ping + 等 SessionStarted）", variant_current),
    ("方案1：去掉 ping", variant_no_ping),
    ("方案2：去 ping + 流水线发帧", variant_pipelined),
]


async def main():
    if not APP_ID or not TOKEN:
        print("❌ 未配置 VOLC_APP_ID / VOLC_ACCESS_TOKEN")
        return

    print(f"资源: {RESOURCE} | 音色: {VOICE}")
    print(f"每方案连测 {len(TEXTS)} 句，复用同一条连接\n")
    print("=" * 72)

    summary = []
    for name, fn in VARIANTS:
        print(f"\n{name}")
        ws = await open_conn()
        vals = []
        try:
            for text in TEXTS:
                try:
                    ms = await fn(ws, text)
                    vals.append(ms)
                    print(f"    首包 {ms:.0f}ms")
                except Exception as exc:
                    print(f"    ❌ {type(exc).__name__}: {exc}")
                await asyncio.sleep(0.3)
        finally:
            try:
                await ws.close()
            except Exception:
                pass
        if vals:
            med = statistics.median(vals)
            summary.append((name, med, len(vals)))
            print(f"    → 中位 {med:.0f}ms")
        else:
            summary.append((name, None, 0))

    print("\n" + "=" * 72)
    print(f"{'方案':38s} {'首包中位':>10s} {'相对现状':>12s}")
    print("-" * 72)
    base = summary[0][1]
    for name, med, n in summary:
        if med is None:
            print(f"{name:38s} {'失败':>10s} {'—':>12s}")
        else:
            delta = f"-{base - med:.0f}ms" if base and med < base else (
                f"+{med - base:.0f}ms" if base else "—")
            print(f"{name:38s} {med:>9.0f}ms {delta:>12s}")


if __name__ == "__main__":
    asyncio.run(main())
