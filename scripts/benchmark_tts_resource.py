"""火山 TTS 资源 ID 可用性 + 首包延迟分段探测。

两件事：
1. 探测哪些 resource_id 能在 v3 bidirection 端点上建连（捕获 400 响应体）。
2. 把可用配置的首包延迟拆成 ping / StartSession / 合成三段，
   定位生产 1.56~1.79s 与裸测 1.15s 之间的差值。

复用 src/tools/voice/ark_tts.py 的协议实现，保证与生产路径一致。
只读 .env，不打印密钥。
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
PROD_VOICE = os.getenv("VOLC_TTS_VOICE", "zh_female_vv_uranus_bigtts")

SENTENCES = [
    "您好，今天感觉怎么样？",
    "我们先做个简单的小测试。",
    "别着急，慢慢来就好。",
]

# 候选资源 ID。ASR 侧用的是 volc.seedasr.sauc.duration，说明 1.x 系列走
# volc.* 命名空间；一并试新老两代常见写法。
CANDIDATE_RESOURCES = [
    ("seed-tts-2.0", "当前生产配置"),
    ("seed-tts-1.0", "seed 1.0 猜测"),
    ("volc.service_type.10029", "大模型语音合成 1.x"),
    ("volc.megatts.default", "声音复刻 1.0"),
    ("volc.megatts.concurr", "声音复刻并发版"),
    ("volc.tts.default", "通用 TTS 猜测"),
]


async def open_conn(resource: str):
    """建连，返回 (ws, 建连耗时)。失败时抛出带响应体的异常。"""
    headers = {
        "X-Api-App-Key": APP_ID,
        "X-Api-Access-Key": TOKEN,
        "X-Api-Resource-Id": resource,
        "X-Api-Connect-Id": str(uuid.uuid4()),
    }
    t0 = time.time()
    try:
        ws = await websockets.connect(_WSS_URL, additional_headers=headers, open_timeout=10)
    except websockets.exceptions.InvalidStatus as exc:
        resp = getattr(exc, "response", None)
        detail = ""
        if resp is not None:
            body = b""
            try:
                body = bytes(resp.body or b"")[:300]
            except Exception:
                pass
            # 火山把拒绝原因放在响应头里
            interesting = {
                k: v for k, v in dict(resp.headers).items()
                if k.lower().startswith("x-") or k.lower() == "server"
            }
            detail = f" status={resp.status_code} headers={interesting} body={body!r}"
        raise RuntimeError(f"建连被拒{detail}") from exc

    await ws.send(_frame_no_id(_EVT_START_CONN, b"{}"))
    resp = _parse_server_frame(await asyncio.wait_for(ws.recv(), timeout=10))
    if resp["event"] != _EVT_CONN_STARTED:
        await ws.close()
        raise RuntimeError(f"StartConnection 失败 event={resp['event']} payload={resp.get('payload')}")
    return ws, time.time() - t0


async def synth_segmented(ws, voice: str, text: str) -> dict:
    """模拟生产路径并分段计时：ping → StartSession → 首个音频包。"""
    seg = {}

    # 生产路径里 _ensure_connected() 每次合成前都做一次 ping 探活
    t_ping = time.time()
    pong_waiter = await ws.ping()
    if pong_waiter is not None:
        await asyncio.wait_for(pong_waiter, timeout=2)
    seg["ping_ms"] = (time.time() - t_ping) * 1000

    t0 = time.time()
    sid = str(uuid.uuid4())
    meta = {
        "user": {"uid": "probe"},
        "req_params": {
            "speaker": voice,
            "audio_params": {
                "format": "pcm",
                "sample_rate": 24000,
                "speech_rate": 0,
                "emotion": "neutral",
                "emotion_scale": 3,
            },
        },
    }
    await ws.send(_frame_with_sid(
        _EVT_START_SESSION, sid, json.dumps(meta, ensure_ascii=False).encode()
    ))
    resp = _parse_server_frame(await asyncio.wait_for(ws.recv(), timeout=10))
    if resp["event"] != _EVT_SESSION_STARTED:
        raise RuntimeError(f"会话启动失败 event={resp['event']} payload={resp.get('payload')}")
    seg["start_session_ms"] = (time.time() - t0) * 1000

    t_task = time.time()
    await ws.send(_frame_with_sid(
        _EVT_TASK_REQUEST, sid,
        json.dumps({"req_params": {"text": text}}, ensure_ascii=False).encode()
    ))
    await ws.send(_frame_with_sid(_EVT_FINISH_SESSION, sid, b"{}"))

    first = None
    nbytes = 0
    while True:
        resp = _parse_server_frame(await asyncio.wait_for(ws.recv(), timeout=30))
        if resp["audio"]:
            if first is None:
                first = time.time() - t_task
                seg["synth_ms"] = first * 1000
            nbytes += len(resp["audio"])
        elif resp["event"] == _EVT_SESSION_FINISHED:
            break
        elif resp["event"] == _EVT_SESSION_FAILED:
            raise RuntimeError(f"会话失败 payload={resp.get('payload')}")

    seg["bytes"] = nbytes
    # 生产口径的首包 = ping + StartSession + 合成
    seg["prod_first_pkt_ms"] = seg["ping_ms"] + seg["start_session_ms"] + seg.get("synth_ms", 0)
    # 去掉 ping 后的首包
    seg["no_ping_ms"] = seg["start_session_ms"] + seg.get("synth_ms", 0)
    return seg


async def probe_resource(resource: str, note: str, voice: str) -> dict:
    row = {"resource": resource, "note": note, "voice": voice}
    ws = None
    try:
        ws, connect_s = await open_conn(resource)
        row["connect_s"] = round(connect_s, 3)
        segs = []
        for text in SENTENCES:
            segs.append(await synth_segmented(ws, voice, text))
            await asyncio.sleep(0.3)
        row["ok"] = True
        row["segments"] = [
            {k: (round(v, 1) if isinstance(v, float) else v) for k, v in s.items()}
            for s in segs
        ]
        for key in ("ping_ms", "start_session_ms", "synth_ms", "prod_first_pkt_ms", "no_ping_ms"):
            vals = [s[key] for s in segs if key in s]
            if vals:
                row[f"median_{key}"] = round(statistics.median(vals), 1)
    except Exception as exc:
        row["ok"] = False
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
    return row


async def main():
    if not APP_ID or not TOKEN:
        print("❌ 未配置 VOLC_APP_ID / VOLC_ACCESS_TOKEN")
        return

    print(f"端点: {_WSS_URL}")
    print(f"音色: {PROD_VOICE}（沿用生产音色，先只变资源 ID）\n")
    print("=" * 80)
    print("第一部分：资源 ID 可用性")
    print("=" * 80)

    results = []
    for resource, note in CANDIDATE_RESOURCES:
        print(f"→ {resource:26s} {note} ...", flush=True)
        row = await probe_resource(resource, note, PROD_VOICE)
        results.append(row)
        if row["ok"]:
            print(
                f"   ✅ 建连 {row['connect_s']}s | "
                f"ping {row['median_ping_ms']}ms + "
                f"StartSession {row['median_start_session_ms']}ms + "
                f"合成 {row['median_synth_ms']}ms "
                f"= 生产口径首包 {row['median_prod_first_pkt_ms']}ms"
            )
        else:
            print(f"   ❌ {row['error'][:220]}")
        print()

    ok_rows = [r for r in results if r["ok"]]

    print("=" * 80)
    print("第二部分：可用配置的首包分段（中位数）")
    print("=" * 80)
    if ok_rows:
        print(f"{'资源':26s} {'ping':>8s} {'StartSess':>10s} {'合成':>8s} {'生产口径':>9s} {'去ping':>8s}")
        print("-" * 80)
        for r in ok_rows:
            print(
                f"{r['resource']:26s} "
                f"{r['median_ping_ms']:>7.0f}ms "
                f"{r['median_start_session_ms']:>9.0f}ms "
                f"{r['median_synth_ms']:>7.0f}ms "
                f"{r['median_prod_first_pkt_ms']:>8.0f}ms "
                f"{r['median_no_ping_ms']:>7.0f}ms"
            )
    else:
        print("没有任何资源 ID 可用")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_resource_probe_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\n明细已写入 {out}")


if __name__ == "__main__":
    asyncio.run(main())
