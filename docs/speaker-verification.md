# 声纹验证模型

当前默认模型为 `iic/speech_eres2netv2_sv_zh-cn_16k-common@v1.0.2`：

- 训练规模：约 20 万名中文说话人
- 输入：16 kHz 单声道音频，80 维 Kaldi Fbank
- 输出：192 维 speaker embedding
- 默认余弦相似度阈值：`0.36`
- 推理：ONNX Runtime CPU，4 个线程

本机稳定态延迟：1 秒音频约 29 ms，2 秒约 50 ms，3 秒约 77 ms。
声纹推理通过工作线程运行，不阻塞 WebSocket 事件循环，并在服务启动时预热。

官方 CN-Celeb EER：ERes2NetV2 `3.81%`、CAM++ `4.32%`、
ECAPA-TDNN `7.45%`、ResNet34 `6.97%`。不同数据和测试协议下的指标不可直接混用。

可通过环境变量切换或回退：

```env
SPEAKER_VERIFIER_MODEL=eres2netv2
SPEAKER_VERIFIER_THRESHOLD=0.36
SPEAKER_VERIFIER_CPU_THREADS=4
```

将 `SPEAKER_VERIFIER_MODEL` 改为 `resnet34` 可回退旧模型。不同模型产生的
embedding 不兼容，切换模型后必须重新录入声纹。

模型来源与评测：

- https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common
- https://github.com/modelscope/3D-Speaker
