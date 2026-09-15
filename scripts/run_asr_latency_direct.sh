#!/usr/bin/env bash
set -euo pipefail

# Run this after disabling Clash Verge TUN / switching to a direct route.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data/luyang/envs/lmca/bin/python}"
RUNS="${RUNS:-20}"
INTERVAL="${INTERVAL:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${OUTPUT:-output/benchmark_ark_asr_streaming_direct_${STAMP}.json}"

if [[ ! -f .env ]]; then
  echo "找不到 .env：$PROJECT_DIR/.env" >&2
  exit 1
fi

# Export project credentials without printing their values.
set -a
# shellcheck disable=SC1091
source .env
set +a

echo "项目目录: $PROJECT_DIR"
echo "测试次数: $RUNS"
echo "输出文件: $OUTPUT"
echo "当前到火山站点的路由（若仍显示 Meta，说明 TUN 还在）："
ASR_IP="$(getent ahostsv4 openspeech.bytedance.com | awk 'NR==1 {print $1}')"
if [[ -n "$ASR_IP" ]]; then
  ip route get "$ASR_IP" 2>/dev/null | head -n 2 || true
fi
echo

exec "$PYTHON_BIN" -u scripts/benchmark_ark_asr_streaming.py \
  --audio tests/fixtures/bench_speech_zh_8s.wav \
  --chunk-ms 100 \
  --realtime \
  --runs "$RUNS" \
  --interval "$INTERVAL" \
  --output "$OUTPUT"
