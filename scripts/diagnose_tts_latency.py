"""精细定位火山 TTS 首包延迟的构成。

上一轮测出 ping 481ms、StartSession 481ms，但网络 RTT 只有 4.4ms，
两段又完全相等——高度怀疑是测量假象。本脚本分三组对照：

  A. 全新连接、未合成过 → ping RTT
  B. 合成完成、连接已排空 → ping RTT
  C. 连续 ping 多次 → 看是否首次特别慢

同时用原始 socket 读取 400 响应体，拿到资源 ID 被拒的真实原因。
"""
import asyncio
import json
import os
import ssl
import statistics
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import websockets
from dotenv import load_dotenv

from src.tools.voice.ark_tts import (
    _WSS_URL,
    _EVT_START_CONN,
    _EVT_CONN_STARTED,
    _EVT_START_SESSION,
    _EVT_SESSION_STARTED,
    _EVT_SESSION_FINISHED,
    _EVT_SESSION_FAILED,
    _EVT_TASK_REQUEST,
    _EVT_FINISH_SESSION,
    _frame_no_id,
    _frame_with_sid,
    _parse_server_frame,
)

load_dotenv()

APP_ID = os.getenv("VOLC_APP_ID", "")
TOKEN = os.getenv("VOLC_ACCESS_TOKEN", "")
VOICE = os.getenv("VOLC_TTS_VOICE", "zh_female_vv_uranus_bigtts")
RESOURCE = os.getenv("VOLC_TTS_RESOURCE", "seed-tts-2.0")

TEXT = "您好，今天感觉怎么样？"


async def timed_ping(ws, label: str) -> float:
    t = time.time()
    waiter = await ws.ping()
    if waiter is not None:
        await asyncio.wait_for(waiter, timeout=5)
    ms = (time.time() - t) * 1000
    print(f"      {label}: {ms:.1f}ms")
    return ms


async def open_conn(resource: str):
    headers = {
        "X-Api-App-Key": APP_ID,
        "X-Api-Access-Key": TOKEN,
        "X-Api-Resource-Id": resource,
        "X-Api-Connect-Id": str(uuid.uuid4()),
    }
    t0 = time.time()
    ws = await websockets.connect(_WSS_URL, additional_headers=headers, open_timeout=10)
    t_ws = time.time() - t0
    t1 = time.time()
    await ws.send(_frame_no_id(_EVT_START_CONN, b"{}"))
    resp = _parse_server_frame(await asyncio.wait_for(ws.recv(), timeout=10))
    if resp["event"] != _EVT_CONN_STARTED:
        await ws.close()
        raise RuntimeError(f"StartConnection 失败 event={resp['event']}")
    return ws, t_ws, time.time() - t1


async def synth(ws, text: str) -> dict:
    """合成一次，分段计时。"""
    sid = str(uuid.uuid4())
    meta = {
        "user": {"uid": "probe"},
        "req_params": {
            "speaker": VOICE,
            "audio_params": {
                "format": "pcm", "sample_rate": 24000,
                "speech_rate": 0, "emotion": "neutral", "emotion_scale": 3,
            },
        },
    }
    t0 = time.time()
    await ws.send(_frame_with_sid(
        _EVT_START_SESSION, sid, json.dumps(meta, ensure_ascii=False).encode()))
    resp = _parse_server_frame(await asyncio.wait_for(ws.recv(), timeout=10))
    if resp["event"] != _EVT_SESSION_STARTED:
        raise RuntimeError(f"会话启动失败 event={resp['event']}")
    start_session_ms = (time.time() - t0) * 1000

    t1 = time.time()
    await ws.send(_frame_with_sid(
        _EVT_TASK_REQUEST, sid,
        json.dumps({"req_params": {"text": text}}, ensure_ascii=False).encode()))
    await ws.send(_frame_with_sid(_EVT_FINISH_SESSION, sid, b"{}"))

    first_ms = None
    nbytes = 0
    while True:
        resp = _parse_server_frame(await asyncio.wait_for(ws.recv(), timeout=30))
        if resp["audio"]:
            if first_ms is None:
                first_ms = (time.time() - t1) * 1000
            nbytes += len(resp["audio"])
        elif resp["event"] == _EVT_SESSION_FINISHED:
            break
        elif resp["event"] == _EVT_SESSION_FAILED:
            raise RuntimeError(f"会话失败 payload={resp.get('payload')}")
    return {
        "start_session_ms": start_session_ms,
        "synth_first_ms": first_ms,
        "drain_ms": (time.time() - t1) * 1000,
        "bytes": nbytes,
    }


async def raw_probe_status(resource: str) -> str:
    """用原始 TLS socket 发 WebSocket 握手，读回完整 HTTP 响应。"""
    ctx = ssl.create_default_context()
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("openspeech.bytedance.com", 443, ssl=ctx), timeout=10)
    except Exception as exc:
        return f"连接失败 {type(exc).__name__}: {exc}"
    key = "dGhlIHNhbXBsZSBub25jZQ=="
    req = (
        "GET /api/v3/tts/bidirection HTTP/1.1\r\n"
        "Host: openspeech.bytedance.com\r\n"
        "Upgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
        f"X-Api-App-Key: {APP_ID}\r\nX-Api-Access-Key: {TOKEN}\r\n"
        f"X-Api-Resource-Id: {resource}\r\n"
        f"X-Api-Connect-Id: {uuid.uuid4()}\r\n\r\n"
    )
    writer.write(req.encode())
    await writer.drain()
    try:
        data = await asyncio.wait_for(reader.read(2048), timeout=10)
    except Exception as exc:
        data = f"读取失败 {exc}".encode()
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass
    text = data.decode("utf-8", "replace")
    status = text.split("\r\n", 1)[0] if text else "(空)"
    msg = ""
    for line in text.split("\r\n"):
        low = line.lower()
        if low.startswith(("x-tt-logid", "x-api-", "server-timing")) or "message" in low:
            msg += f"\n        {line}"
    body = text.split("\r\n\r\n", 1)[1][:200] if "\r\n\r\n" in text else ""
    return f"{status}{msg}" + (f"\n        body={body!r}" if body.strip() else "")


async def main():
    if not APP_ID or not TOKEN:
        print("❌ 未配置 VOLC_APP_ID / VOLC_ACCESS_TOKEN")
        return

    print("=" * 78)
    print("第一部分：ping 481ms 是真的吗")
    print("=" * 78)

    ws, t_ws, t_startconn = await open_conn(RESOURCE)
    print(f"\n  建连: WebSocket握手 {t_ws*1000:.0f}ms + StartConnection {t_startconn*1000:.0f}ms")

    print("\n  A. 全新连接、未合成过:")
    fresh = [await timed_ping(ws, f"ping#{i+1}") for i in range(3)]

    print("\n  B. 合成一次后（连接已排空）:")
    s1 = await synth(ws, TEXT)
    print(f"      合成: StartSession {s1['start_session_ms']:.0f}ms + "
          f"首音频 {s1['synth_first_ms']:.0f}ms (排空 {s1['drain_ms']:.0f}ms, {s1['bytes']}B)")
    after = [await timed_ping(ws, f"ping#{i+1}") for i in range(3)]

    print("\n  C. 再合成两次，看 StartSession 是否稳定:")
    later = []
    for i in range(2):
        s = await synth(ws, TEXT)
        later.append(s)
        print(f"      #{i+1}: StartSession {s['start_session_ms']:.0f}ms + "
              f"首音频 {s['synth_first_ms']:.0f}ms")
    await ws.close()

    print("\n" + "-" * 78)
    print(f"  全新连接 ping 中位: {statistics.median(fresh):.1f}ms")
    print(f"  合成后   ping 中位: {statistics.median(after):.1f}ms")
    all_syn = [s1] + later
    print(f"  StartSession 中位 : {statistics.median([s['start_session_ms'] for s in all_syn]):.1f}ms")
    print(f"  合成首音频 中位   : {statistics.median([s['synth_first_ms'] for s in all_syn]):.1f}ms")

    print("\n" + "=" * 78)
    print("第二部分：其他资源 ID 被拒的真实原因")
    print("=" * 78)
    for res in ["seed-tts-2.0", "seed-tts-1.0", "volc.service_type.10029",
                "volc.megatts.default", "volc.tts.default"]:
        print(f"\n  {res}:")
        print(f"      {await raw_probe_status(res)}")


if __name__ == "__main__":
    asyncio.run(main())
