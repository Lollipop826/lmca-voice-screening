"""ArkASR 预热连接复用机制的单元测试（不依赖火山引擎，用本地 mock WS 服务器）。"""
import asyncio
import gzip
import json
import os
import struct
import unittest

import numpy as np
import websockets

os.environ.setdefault("VOLC_APP_ID", "test-app")
os.environ.setdefault("VOLC_ACCESS_TOKEN", "test-token")
os.environ["ARK_ASR_PREWARM"] = "true"
# 本地 mock 服务器不能走系统 HTTP 代理，否则 websockets 16.x 默认 proxy=True 会拦截握手
for _pk in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_pk, None)

from src.tools.voice import ark_asr


def _server_last_frame(text: str) -> bytes:
    """构造一个 gzip+JSON 的“最后包”服务端响应帧（msg_type=0b1001, flags=last）。"""
    payload = gzip.compress(json.dumps({"result": {"text": text}}).encode())
    hdr = bytes([0x91, 0x02, 0x11, 0x00])  # 0b1001<<4 | 0b0010(last); serial=JSON,gzip
    return hdr + struct.pack(">I", len(payload)) + payload


class ArkASRPrewarmTests(unittest.TestCase):
    def setUp(self):
        # 每个用例重置全局预热状态
        ark_asr._warm_socket = None
        ark_asr._warm_fail_streak = 0
        ark_asr._warm_disabled = False
        ark_asr._warm_refill_task = None

    def _run_server(self, text, connections_holder):
        async def handler(ws):
            connections_holder.append(ws)
            try:
                async for _ in ws:
                    # 收到任意音频/config 帧后，回一个最终结果帧
                    await ws.send(_server_last_frame(text))
                    break
            except websockets.ConnectionClosed:
                pass
        return handler

    def test_prewarm_socket_is_reused(self):
        async def scenario():
            conns = []
            server = await websockets.serve(self._run_server("你好世界", conns), "127.0.0.1", 0)
            port = server.sockets[0].getsockname()[1]
            ark_asr._WSS_URL = f"ws://127.0.0.1:{port}"

            audio = np.zeros(16000, dtype=np.float32)
            try:
                # 手动预建一条 warm socket
                await ark_asr._refill_warm_socket()
                self.assertIsNotNone(ark_asr._warm_socket, "预热连接应已建立")

                result = await ark_asr.ark_asr_recognize(audio, sample_rate=16000)
                self.assertEqual(result["text"], "你好世界")
                # 预热成功后应调度补充下一条；给它一点时间完成
                await asyncio.sleep(0.2)
            finally:
                server.close()
                await server.wait_closed()
                if ark_asr._warm_socket is not None:
                    await ark_asr._warm_socket.aclose()

        asyncio.run(scenario())

    def test_falls_back_when_no_warm_socket(self):
        async def scenario():
            conns = []
            server = await websockets.serve(self._run_server("回退成功", conns), "127.0.0.1", 0)
            port = server.sockets[0].getsockname()[1]
            ark_asr._WSS_URL = f"ws://127.0.0.1:{port}"

            audio = np.zeros(16000, dtype=np.float32)
            try:
                # 没有预建连接，应走新建路径
                self.assertIsNone(ark_asr._warm_socket)
                result = await ark_asr.ark_asr_recognize(audio, sample_rate=16000)
                self.assertEqual(result["text"], "回退成功")
                await asyncio.sleep(0.2)
            finally:
                server.close()
                await server.wait_closed()
                if ark_asr._warm_socket is not None:
                    await ark_asr._warm_socket.aclose()

        asyncio.run(scenario())

    def test_expired_warm_socket_is_discarded(self):
        async def scenario():
            conns = []
            server = await websockets.serve(self._run_server("新建路径", conns), "127.0.0.1", 0)
            port = server.sockets[0].getsockname()[1]
            ark_asr._WSS_URL = f"ws://127.0.0.1:{port}"
            try:
                await ark_asr._refill_warm_socket()
                self.assertIsNotNone(ark_asr._warm_socket)
                # 强制过期
                ark_asr._warm_socket._created_at -= (ark_asr._WARM_TTL + 5)
                got = await ark_asr._get_warm_socket()
                self.assertIsNone(got, "过期连接应被丢弃")
            finally:
                server.close()
                await server.wait_closed()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
