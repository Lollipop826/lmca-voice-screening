#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$PROJECT_DIR/deploy/docker-compose.cloudflare.yml"
TOKEN_FILE="$PROJECT_DIR/.env.cloudflare"
ACTION="${1:-up}"

if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE=(docker-compose)
else
    echo "错误：没有找到 docker compose 或 docker-compose。" >&2
    exit 1
fi

compose() {
    "${COMPOSE[@]}" -p voice-cloudflare-tunnel -f "$COMPOSE_FILE" "$@"
}

require_token() {
    if [ ! -f "$TOKEN_FILE" ]; then
        echo "错误：缺少 $TOKEN_FILE" >&2
        echo "请复制 .env.cloudflare.example，并填入 Cloudflare Tunnel Token。" >&2
        exit 1
    fi
    if ! grep -Eq '^TUNNEL_TOKEN=.+$' "$TOKEN_FILE" || grep -Eq '^TUNNEL_TOKEN=replace-' "$TOKEN_FILE"; then
        echo "错误：$TOKEN_FILE 中的 TUNNEL_TOKEN 尚未配置。" >&2
        exit 1
    fi
}

require_origin() {
    if ! curl --fail --silent --show-error --max-time 5 \
        http://127.0.0.1:8502/health >/dev/null; then
        echo "错误：本地服务 http://127.0.0.1:8502/health 尚未就绪。" >&2
        echo "请先运行：./start_voice_only.sh" >&2
        exit 1
    fi
}

case "$ACTION" in
    up)
        require_token
        require_origin
        compose up -d
        echo "Cloudflare Tunnel 已启动。"
        echo "查看状态：deploy/cloudflare-tunnel.sh status"
        echo "查看日志：deploy/cloudflare-tunnel.sh logs"
        ;;
    down)
        compose down
        ;;
    restart)
        require_token
        require_origin
        compose restart
        ;;
    status)
        compose ps
        ;;
    logs)
        compose logs --tail=100 -f
        ;;
    *)
        echo "用法：$0 {up|down|restart|status|logs}" >&2
        exit 2
        ;;
esac
