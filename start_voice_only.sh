#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
TMP_DIR="$PROJECT_DIR/tmp"
ENV_FILE="$PROJECT_DIR/.env"
mkdir -p "$TMP_DIR"

config_value() {
    local name="$1"
    local default="$2"
    local value="$(printenv "$name" 2>/dev/null || true)"
    if [ -z "$value" ] && [ -f "$ENV_FILE" ]; then
        value="$(awk -F= -v key="$name" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' "$ENV_FILE")"
    fi
    value="${value:-$default}"
    value="${value#\"}"
    value="${value%\"}"
    printf '%s' "$value"
}

is_true() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

port_is_ready() {
    local host="$1"
    local port="$2"
    "$PYTHON_BIN" - "$host" "$port" <<'PY'
import socket
import sys

try:
    with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=0.5):
        pass
except OSError:
    raise SystemExit(1)
PY
}

port_is_listening() {
    command -v ss >/dev/null 2>&1 && ss -ltnH | grep -Eq ":${1}[[:space:]]"
}

wait_for_port() {
    local host="$1"
    local port="$2"
    local timeout="$3"
    for ((i = 0; i < timeout; i++)); do
        port_is_ready "$host" "$port" && return 0
        sleep 1
    done
    return 1
}

json_value() {
    local body="$1"
    local path="$2"
    printf '%s' "$body" | "$PYTHON_BIN" -c '
import json, sys
value = json.load(sys.stdin)
for part in sys.argv[1].split("."):
    value = value.get(part) if isinstance(value, dict) else None
print("" if value is None else str(value).lower() if isinstance(value, bool) else value)
' "$path"
}

health_request() {
    curl -ksS --max-time 3 "https://127.0.0.1:${VOICE_PORT}/health" 2>/dev/null || \
        curl -fsS --max-time 3 "http://127.0.0.1:${VOICE_PORT}/health" 2>/dev/null || true
}

stop_project_pid() {
    local pid="$1"
    local expected="$2"
    local cmd pid_cwd parent_pid parent_cmd parent_cwd
    [[ "$pid" =~ ^[0-9]+$ ]] || return 0
    kill -0 "$pid" 2>/dev/null || return 0
    cmd="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
    pid_cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
    if [[ "$pid_cwd" != "$PROJECT_DIR" || "$cmd" != *"$expected"* ]]; then
        echo "忽略不属于本项目的 PID=$pid"
        return 0
    fi
    parent_pid="$(awk '/^PPid:/{print $2}' "/proc/$pid/status" 2>/dev/null || true)"
    if [[ "$parent_pid" =~ ^[0-9]+$ && "$parent_pid" != "1" && "$parent_pid" != "$$" ]] && kill -0 "$parent_pid" 2>/dev/null; then
        parent_cmd="$(tr '\0' ' ' <"/proc/$parent_pid/cmdline" 2>/dev/null || true)"
        parent_cwd="$(readlink -f "/proc/$parent_pid/cwd" 2>/dev/null || true)"
        if [[ "$parent_cwd" == "$PROJECT_DIR" && "$parent_cmd" == *"start_voice_only.sh"* ]]; then
            echo "停止旧的本项目启动脚本 PID=$parent_pid"
            kill "$parent_pid" 2>/dev/null || true
            for ((i = 0; i < 20; i++)); do
                kill -0 "$parent_pid" 2>/dev/null || break
                sleep 0.5
            done
        fi
    fi
    echo "停止旧的本项目进程 PID=$pid"
    kill "$pid" 2>/dev/null || true
    for ((i = 0; i < 20; i++)); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
}

stop_recorded_process() {
    local pid_file="$1"
    local expected="$2"
    [ -f "$pid_file" ] || return 0
    stop_project_pid "$(cat "$pid_file" 2>/dev/null || true)" "$expected"
    rm -f "$pid_file"
}

stop_port_processes() {
    local port="$1"
    local pid
    while read -r pid; do
        [ -n "$pid" ] || continue
        stop_project_pid "$pid" voice_server.py
    done < <(
        ss -ltnpH 2>/dev/null \
            | awk -v port="$port" '$4 ~ (":" port "$") {print}' \
            | grep -oE 'pid=[0-9]+' \
            | cut -d= -f2 \
            | sort -u
    )
}

VOICE_PORT="$(config_value VOICE_PORT 8426)"
PYTHON_BIN="$(config_value PYTHON_BIN /home/student1/miniforge3/envs/lmca/bin/python)"
VOICE_PID_FILE="$TMP_DIR/voice_server_${VOICE_PORT}.pid"
VOICE_LOG="$TMP_DIR/voice_server.log"
SOULX_PID_FILE="$TMP_DIR/soulx_turn.pid"
SOULX_LOG="$TMP_DIR/soulx_turn.log"
VOICE_STARTED_HERE=0
SOULX_STARTED_HERE=0
SOULX_PID=""

if [ ! -x "$PYTHON_BIN" ]; then
    echo "错误：找不到 LMCA Python 环境: $PYTHON_BIN" >&2
    exit 1
fi

SOULX_ENABLED="$(config_value USE_SOULX_TURN_TAKING true)"
SOULX_DIR="$(config_value SOULX_DIR /data/student1/SoulX-Duplug)"
SOULX_PYTHON="$(config_value SOULX_PYTHON /data/student1/conda-envs/soulx-duplug/bin/python)"
SOULX_HOST="$(config_value SOULX_HOST 127.0.0.1)"
SOULX_PORT="$(config_value SOULX_PORT 8001)"
SOULX_DEVICE="$(config_value SOULX_DEVICE cpu)"
SOULX_CUDA_VISIBLE_DEVICES="$(config_value SOULX_CUDA_VISIBLE_DEVICES 0)"
SOULX_MODELSCOPE_CACHE="$(config_value MODELSCOPE_CACHE /data/student1/yzy-cache/modelscope)"


export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/ZipVoice-master:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="/usr/local/cuda-11.8"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="$(config_value HF_HOME /home/student1/.cache/huggingface)"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cleanup() {
    if [ "$VOICE_STARTED_HERE" = 1 ] && [ -n "${VOICE_PID:-}" ]; then
        kill "$VOICE_PID" 2>/dev/null || true
        wait "$VOICE_PID" 2>/dev/null || true
    fi
    if [ "$SOULX_STARTED_HERE" = 1 ] && [ -n "$SOULX_PID" ]; then
        kill "$SOULX_PID" 2>/dev/null || true
        wait "$SOULX_PID" 2>/dev/null || true
        rm -f "$SOULX_PID_FILE"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "======================================"
echo "启动 LMCA 完整语音系统 (端口 $VOICE_PORT)"
echo "======================================"

stop_recorded_process "$VOICE_PID_FILE" voice_server.py
stop_port_processes "$VOICE_PORT"

if port_is_listening "$VOICE_PORT"; then
    echo "错误：端口 $VOICE_PORT 已被其他进程占用，未停止该进程。" >&2
    exit 1
fi


if is_true "$SOULX_ENABLED"; then
    echo "检查 SoulX 轮次服务 ($SOULX_HOST:$SOULX_PORT)..."
    if port_is_ready "$SOULX_HOST" "$SOULX_PORT"; then
        echo "✅ SoulX 已在运行，直接复用"
    else
        if [ ! -f "$SOULX_DIR/server.py" ] || [ ! -x "$SOULX_PYTHON" ]; then
            echo "错误：SoulX 已启用但目录或 Python 不存在。" >&2
            exit 1
        fi
        (
            cd "$SOULX_DIR"
            CUDA_VISIBLE_DEVICES="$SOULX_CUDA_VISIBLE_DEVICES" \
            SOULX_DEVICE="$SOULX_DEVICE" \
            MODELSCOPE_CACHE="$SOULX_MODELSCOPE_CACHE" \
            HF_ENDPOINT="$HF_ENDPOINT" \
            "$SOULX_PYTHON" -u -m uvicorn server:app \
                --host "$SOULX_HOST" --port "$SOULX_PORT" --workers 1
        ) >"$SOULX_LOG" 2>&1 &
        SOULX_PID=$!
        SOULX_STARTED_HERE=1
        echo "$SOULX_PID" >"$SOULX_PID_FILE"
        if ! wait_for_port "$SOULX_HOST" "$SOULX_PORT" 180; then
            echo "错误：SoulX 启动超时，日志: $SOULX_LOG" >&2
            tail -n 50 "$SOULX_LOG" >&2 || true
            exit 1
        fi
        echo "✅ SoulX 已就绪 (PID=$SOULX_PID)"
    fi
else
    echo "⏭️ SoulX：按配置关闭，使用本地 Silero VAD 全双工"
fi

MEMOBASE_URL="$(config_value MEMOBASE_PROJECT_URL '')"
if [ -n "$MEMOBASE_URL" ]; then
    if ! curl -fsS --max-time 3 "$MEMOBASE_URL/api/v1/healthcheck" >/dev/null 2>&1; then
        echo "错误：MEMOBASE_PROJECT_URL 已配置但服务不可用: $MEMOBASE_URL" >&2
        exit 1
    fi
    echo "✅ Memobase API 已就绪 ($MEMOBASE_URL)；SQLite 仍是本地事实库"
else
    echo "✅ 记忆服务：SQLite 本地引擎（内置 worker，无需单独进程）"
fi

echo "启动主服务，等待模型和依赖预热..."
"$PYTHON_BIN" -u "$PROJECT_DIR/voice_server.py" >>"$VOICE_LOG" 2>&1 &
VOICE_PID=$!
VOICE_STARTED_HERE=1
echo "$VOICE_PID" >"$VOICE_PID_FILE"

STARTUP_TIMEOUT="$(config_value STARTUP_TIMEOUT_S 300)"
READY_BODY=""
for ((i = 0; i < STARTUP_TIMEOUT; i++)); do
    if ! kill -0 "$VOICE_PID" 2>/dev/null; then
        echo "错误：主服务提前退出，日志: $VOICE_LOG" >&2
        tail -n 80 "$VOICE_LOG" >&2 || true
        exit 1
    fi
    READY_BODY="$(health_request)"
    if [ -n "$READY_BODY" ] && [ "$(json_value "$READY_BODY" status)" = ok ] && \
        [ "$(json_value "$READY_BODY" startup.ready)" = true ]; then
        break
    fi
    sleep 1
done

if [ -z "$READY_BODY" ] || [ "$(json_value "$READY_BODY" startup.ready 2>/dev/null || true)" != true ]; then
    echo "错误：主服务在 ${STARTUP_TIMEOUT}s 内未完成启动/预热，日志: $VOICE_LOG" >&2
    tail -n 100 "$VOICE_LOG" >&2 || true
    exit 1
fi

echo "✅ 所有已启用服务和预热已就绪："
printf '%s\n' "$READY_BODY"
echo "服务保持运行；停止请终止本脚本。"

wait "$VOICE_PID"
