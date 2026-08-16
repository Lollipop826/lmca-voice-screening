#!/bin/bash

# 获取当前脚本所在目录（自动适配路径）
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 日志目录
TMP_DIR="$PROJECT_DIR/tmp"

# SoulX-Duplug 只作为 8502 的轮次判断模块，不启动它自己的 55556 对话页面。
SOULX_DIR="${SOULX_DIR:-/home/luy/luyang/pause/SoulX-Duplug-main}"
SOULX_PYTHON="${SOULX_PYTHON:-/home/luy/miniconda3/envs/soulx-duplug-repro/bin/python}"
SOULX_HOST="${SOULX_HOST:-127.0.0.1}"
SOULX_PORT="${SOULX_PORT:-8000}"
SOULX_CUDA_VISIBLE_DEVICES="${SOULX_CUDA_VISIBLE_DEVICES:-1}"
SOULX_ENABLED="${USE_SOULX_TURN_TAKING:-true}"
SOULX_PID=""
SOULX_STARTED_HERE=0

# 加载存储配置
if [ -f "$PROJECT_DIR/.env.storage" ]; then
    source "$PROJECT_DIR/.env.storage"
    echo "✅ 已加载外部存储配置"
fi

echo "======================================"
echo "🎤 启动 WebRTC 语音服务 (端口 8502)"
echo "======================================"

# 清理旧进程
echo "🔪 清理旧进程..."
pkill -9 -f voice_server.py 2>/dev/null
sleep 2
echo "✅ 旧进程已清理"

# 进入项目目录
cd "$PROJECT_DIR" || exit

# 启动 WebRTC 服务
echo ""
echo "🚀 启动 WebRTC 服务..."

mkdir -p "$TMP_DIR"

# 设置环境变量
export PYTHONPATH="${PWD}:${PWD}/ZipVoice-master:${PYTHONPATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
export CUDA_HOME="/usr/local/cuda-11.8"
export HF_ENDPOINT="https://hf-mirror.com"

# HuggingFace 缓存目录指向当前用户，避免访问 /root 路径
export HF_HOME="/home/luy/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"

# 解决 k2 与其他 CUDA 模型的内存冲突
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 自动寻找 python，优先使用项目内环境
if [ -x "$PROJECT_DIR/luyang/bin/python" ]; then
    PYTHON_BIN="$PROJECT_DIR/luyang/bin/python"
else
    PYTHON_BIN=$(which python3 || which python)
fi

echo "🐍 使用 Python: $PYTHON_BIN"

# 未从外部显式传入时，从 .env 读取开关；不 source 整个密钥文件。
if [ -z "${USE_SOULX_TURN_TAKING+x}" ] && [ -f "$PROJECT_DIR/.env" ]; then
    SOULX_ENABLED=$("$PYTHON_BIN" - "$PROJECT_DIR/.env" <<'PY'
import sys
from dotenv import dotenv_values

value = dotenv_values(sys.argv[1]).get("USE_SOULX_TURN_TAKING", "true")
print(str(value or "true").strip())
PY
    )
fi

port_is_ready() {
    "$PYTHON_BIN" - "$1" "$2" <<'PY'
import socket
import sys

try:
    with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=0.5):
        pass
except OSError:
    raise SystemExit(1)
PY
}

cleanup_soulx() {
    if [ "$SOULX_STARTED_HERE" = "1" ] && [ -n "$SOULX_PID" ]; then
        echo ""
        echo "🧠 正在停止本次启动的 SoulX 轮次服务 (PID: $SOULX_PID)..."
        kill "$SOULX_PID" 2>/dev/null || true
        wait "$SOULX_PID" 2>/dev/null || true
    fi
}
trap cleanup_soulx EXIT INT TERM

if [ "${SOULX_ENABLED,,}" = "true" ]; then
    echo ""
    echo "🧠 检查 SoulX 全双工轮次服务 (${SOULX_HOST}:${SOULX_PORT})..."
    if port_is_ready "$SOULX_HOST" "$SOULX_PORT"; then
        echo "✅ SoulX 已在运行，直接复用"
    else
        if [ ! -f "$SOULX_DIR/server.py" ]; then
            echo "❌ 找不到 SoulX 服务: $SOULX_DIR/server.py"
            exit 1
        fi
        if [ ! -x "$SOULX_PYTHON" ]; then
            echo "❌ 找不到 SoulX Python 环境: $SOULX_PYTHON"
            exit 1
        fi

        echo "🚀 启动 SoulX（GPU ${SOULX_CUDA_VISIBLE_DEVICES}，首次约需 30-60 秒）..."
        (
            cd "$SOULX_DIR" || exit 1
            CUDA_VISIBLE_DEVICES="$SOULX_CUDA_VISIBLE_DEVICES" \
            HF_ENDPOINT="$HF_ENDPOINT" \
            "$SOULX_PYTHON" -u -m uvicorn server:app \
                --host "$SOULX_HOST" --port "$SOULX_PORT" --workers 1
        ) >"$TMP_DIR/soulx_turn.log" 2>&1 &
        SOULX_PID=$!
        SOULX_STARTED_HERE=1
        echo "$SOULX_PID" > "$TMP_DIR/soulx_turn.pid"

        SOULX_READY=0
        for _ in $(seq 1 120); do
            if port_is_ready "$SOULX_HOST" "$SOULX_PORT"; then
                SOULX_READY=1
                break
            fi
            if ! kill -0 "$SOULX_PID" 2>/dev/null; then
                echo "❌ SoulX 启动失败，最近日志："
                tail -n 40 "$TMP_DIR/soulx_turn.log"
                exit 1
            fi
            sleep 1
        done

        if [ "$SOULX_READY" != "1" ]; then
            echo "❌ SoulX 启动超时，查看日志: $TMP_DIR/soulx_turn.log"
            exit 1
        fi
        echo "✅ SoulX 轮次服务已就绪 (PID: $SOULX_PID)"
    fi
fi

$PYTHON_BIN -u voice_server.py 2>&1 | tee "$TMP_DIR/voice_server.log"

echo ""
echo "❌ 服务已停止"
