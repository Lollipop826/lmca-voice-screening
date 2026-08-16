# ============================================
# 智能语音对话系统 - 轻量版（API-only，无 GPU）
# 适用于阿里云 ECS 等普通云服务器
# ============================================
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app"
ENV HF_ENDPOINT="https://hf-mirror.com"

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 安装 Python 依赖（利用 Docker 层缓存）
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com \
    --timeout 120

COPY requirements-emotion2vec.txt /app/requirements-emotion2vec.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1+cpu torchaudio==2.4.1+cpu && \
    pip install --no-cache-dir -r requirements-emotion2vec.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com \
    --timeout 120

# 使用仓库内已验证的 Silero VAD ONNX 模型，避免构建时拉取无关 GPU 依赖
COPY models/silero_vad.onnx /app/models/silero_vad.onnx

# 复制项目代码
COPY voice_server.py /app/
COPY src/ /app/src/
COPY static/ /app/static/
COPY .env.example /app/.env.example

# 创建临时目录
RUN mkdir -p /app/tmp

# 暴露端口
EXPOSE 8502

# 启动命令
CMD ["python", "-m", "uvicorn", "voice_server:app", "--host", "0.0.0.0", "--port", "8502"]
