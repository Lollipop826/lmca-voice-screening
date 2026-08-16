# Cloudflare 公网部署指南

这个项目是一个 `FastAPI + uvicorn` 语音服务，主入口是 `voice_server:app`，默认监听 `8502`。由于服务依赖 Python 运行时、WebSocket、音频处理，并且可能依赖本机/GPU 侧的 SoulX 轮次服务，最稳的 Cloudflare 部署方式是：

> 项目继续运行在你的云主机或本机，Cloudflare Tunnel 提供 HTTPS 公网入口。

## 推荐架构

```text
用户浏览器
  ↓ HTTPS / WebSocket
Cloudflare
  ↓ Tunnel
cloudflared
  ↓ http://127.0.0.1:8502
voice_server.py
```

## 你需要准备

- 一个已接入 Cloudflare 的域名，例如 `example.com`
- 一台能长期运行项目的机器：本机、VPS、GPU 云主机都可以
- 已安装 Docker（项目也支持直接安装 `cloudflared`，但本文优先使用 Docker）
- 项目的 `.env` 已配置好 API Key、ASR/TTS、SoulX 等开关

## 1. 先让服务本地跑起来

在项目根目录执行：

```bash
./start_voice_only.sh
```

确认本机能打开：

```text
http://127.0.0.1:8502
```

如果云主机上没有 SoulX/GPU 服务，先在 `.env` 里关闭：

```env
USE_SOULX_TURN_TAKING=false
ENABLE_FULL_DUPLEX=false
```

## 2. 在 Cloudflare 控制台创建 Tunnel（推荐）

进入 Cloudflare 控制台：

1. 打开 **Zero Trust / Networking / Tunnels**；
2. 选择 **Create a tunnel**，类型选 `cloudflared`；
3. 名称可填写 `soulx-voice`；
4. 在 **Install connector** 页面选择 Docker，复制命令里 `--token` 后面的 token；
5. 在 **Published application routes** 添加公网路由：
   - Subdomain：例如 `voice`
   - Domain：你的 Cloudflare 域名，例如 `example.com`
   - Service type：`HTTP`
   - URL：`127.0.0.1:8502`

不要在 URL 中填写公网域名，也不要填写 `https://127.0.0.1:8502`；本地 uvicorn
提供的是 HTTP，公网 HTTPS 由 Cloudflare 终止。

## 3. 保存 Tunnel Token

在项目根目录执行：

```bash
cp .env.cloudflare.example .env.cloudflare
chmod 600 .env.cloudflare
```

编辑 `.env.cloudflare`，只把 token 写到等号后：

```env
TUNNEL_TOKEN=你的真实token
```

`.env.cloudflare` 已被 `.gitignore` 排除，不会被正常提交到 Git。

## 4. 启动生产 Tunnel

```bash
deploy/cloudflare-tunnel.sh up
```

检查状态和日志：

```bash
deploy/cloudflare-tunnel.sh status
deploy/cloudflare-tunnel.sh logs
```

Tunnel 容器使用 `restart: unless-stopped`，Docker 服务重启后会自动恢复连接。

浏览器打开：

```text
https://voice.example.com
```

登录页和语音 WebSocket 都走同一个域名，不需要额外设置 WebSocket 端口。

## 5. 让应用本身常驻

Tunnel 只负责公网入口，`voice_server.py` 也必须一直运行。测试阶段可以使用：

```bash
./start_voice_only.sh
```

正式服务器建议再用 `systemd` 或其他进程管理器托管该脚本，并保证机器不会休眠。
如果这是一台个人电脑，关机、休眠或断网后，公网服务都会暂时不可用。

## 命令行创建 Tunnel（备选）

如果你已经在主机上安装了 `cloudflared`，也可以采用本地管理方式：

登录 Cloudflare：

```bash
cloudflared tunnel login
```

创建 tunnel：

```bash
cloudflared tunnel create soulx-voice
```

把域名指到 tunnel，例如：

```bash
cloudflared tunnel route dns soulx-voice voice.example.com
```

### 写入 tunnel 配置

复制模板：

```bash
mkdir -p ~/.cloudflared
cp deploy/cloudflared-config.example.yml ~/.cloudflared/config.yml
```

然后修改 `~/.cloudflared/config.yml`：

- `tunnel` 改成 `cloudflared tunnel create` 返回的 UUID
- `credentials-file` 改成对应 JSON 凭据路径
- `hostname` 改成你的公网域名，例如 `voice.example.com`

### 前台测试

```bash
cloudflared tunnel run soulx-voice
```

浏览器打开：

```text
https://voice.example.com
```

登录页和语音 WebSocket 都应该走同一个域名。

### 后台常驻

确认前台测试成功后，把 tunnel 安装成系统服务：

```bash
sudo cloudflared service install
sudo systemctl enable --now cloudflared
sudo systemctl status cloudflared
```

你的 `voice_server.py` 也需要常驻。可以用 `systemd`、`tmux`、`pm2` 或 Docker 管理。

## 对外开放前检查

如果你想“所有知道链接的人都能访问”：

- 不要给这个域名启用 Cloudflare Access 登录策略
- 保留项目自己的登录/注册逻辑
- 在 Cloudflare DNS 中确认 `voice.example.com` 是橙云代理状态
- 当前应用允许访客自行注册；公开后会产生真实的 ASR、TTS 和 LLM API 调用费用
- 建议在 Cloudflare WAF 中给登录、注册和语音接口设置合理的限速规则
- 不要公开 `.env`、Tunnel Token、API Key 或 `data/.auth_secret`

如果你想只给团队访问：

- 在 Cloudflare Zero Trust 里给 `voice.example.com` 添加 Access Application
- 用邮箱、Google Workspace、GitHub 等身份源限制访问

## 为什么不直接用 Pages / Containers

Cloudflare Pages 只适合静态前端，不能直接运行这个 Python/FastAPI 服务。Cloudflare
Containers 可以运行 Docker 镜像，但当前项目的本地模型目录约 26 GB，超过现有单实例
20 GB 磁盘上限，并且 SoulX 需要本机 GPU，所以现阶段优先使用 Tunnel。

如果后续要迁移到 Containers，需要额外增加：

- 一个 Worker 作为入口
- `wrangler.toml`
- Container class 转发到容器内 `8502`
- 对模型体积、冷启动、持久化数据、密钥和付费计划做专项验证
