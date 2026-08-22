# 交接包清单

> 历史快照，记录 2026-08-13 的交接包范围；不代表当前 Git 仓库的实际文件状态。

生成日期：2026-08-13  
包用途：源码交接与后续开发。除明确授权的“林秀兰”虚构演示测试数据外，不包含本机密钥、其他患者数据、其他运行数据或模型权重。

## 包含

- `voice_server.py`、`src/`、`tests/`、`scripts/`
- `frontend/` 源码、`package.json`、`package-lock.json`、Vite 配置；不含 `node_modules/` 和 `dist/`
- `config/`、`deploy/`、`docs/`、`static/`（不含 `static/audio/`）
- `kb/` 知识库切片
- `vendor/memobase/` Memobase 源码（去除 Python 缓存）
- `ZipVoice-master/` 运行源码（不含模型权重和缓存）
- 根目录入口、依赖、Docker、产品/设计/交付文档
- `docs/archive/handover-2026-08-13.md`、历史实验记录和归档资料
- `data/voice_server.db`：只保留虚构演示患者“林秀兰”（`pt_a71f682f92f74f7d`）及其关联记录的脱敏测试副本
- `data/voice_calls/`：只保留林秀兰对应的 5 个测试会话目录及音频
- `docs/test-data-lin-xiulan.md`：测试数据范围、脱敏处理和使用说明

## 排除

- `.env`、`.env.storage`、`.env.cloudflare` 和 `data/` 下的密钥文件
- 原始 `data/` 整库、其他患者/会话、`tmp/`、运行日志和 Playwright 临时数据
- `models/`、本地模型权重、下载缓存
- `frontend/node_modules/`、`frontend/dist/`
- `__pycache__/`、`*.pyc`、`.pytest_cache/`、`.git/`
- `static/audio/` 本地测试参考音频

## 测试数据例外

“林秀兰”档案的 `extra_profile.fictional_demo=true`，并注明不对应真实个人。包内数据库是从原库按 `patient_id` 重新筛选、脱敏并 `VACUUM` 后生成的测试副本：账号、密码哈希、API key、认证 secret、患者分配和其他患者数据均不包含。详见 `../test-data-lin-xiulan.md`。

## 生成前验证

```text
python -m pytest -q              362 passed in 8.09s
python -m compileall ...         PASS
frontend: npm run build          PASS
docker compose config --quiet    PASS
```

## 接收方启动前

1. 复制 `.env.example` 为 `.env`，自行填写密钥。
2. 按 `../../README.md` 安装 Python 和前端依赖。
3. 先运行测试，再启动服务。
4. 真实语音链路需要云端 ASR/TTS/LLM 凭据；SoulX 是可选外部依赖。
