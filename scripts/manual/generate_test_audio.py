"""
生成测试音频文件（合成语音用于测试）
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tests" / "test_audio"

print("=" * 60)
print("生成测试音频文件")
print("=" * 60)

# 检查是否有 TTS 引擎可用
print("\n【方案1】使用系统内置 TTS 生成测试音频")
print("尝试使用 pyttsx3...")

try:
    import pyttsx3
    import wave
    import numpy as np

    # 初始化 TTS
    engine = pyttsx3.init()

    # 设置中文语音（如果可用）
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'zh' in voice.id.lower() or 'chinese' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            print(f"✅ 找到中文语音: {voice.name}")
            break

    # 生成测试语音
    test_cases = [
        ("happy.wav", "我今天心情很好，非常开心", 150),  # 高音调
        ("anxious.wav", "我最近总是担心这担心那", 100),  # 低音调
        ("calm.wav", "我感觉还可以，比较平静", 120),    # 中等
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, text, rate in test_cases:
        output_path = OUTPUT_DIR / filename
        engine.setProperty('rate', rate)
        engine.save_to_file(text, str(output_path))

    engine.runAndWait()

    print("\n✅ 测试音频生成完成！")
    for filename, text, _ in test_cases:
        path = OUTPUT_DIR / filename
        if path.exists():
            size = path.stat().st_size
            print(f"  {filename}: {text} ({size} bytes)")

except ImportError:
    print("⚠️ pyttsx3 未安装")
    print("\n【方案2】手动准备音频")
    print("请按以下步骤准备测试音频：")
    print("\n1. 录制或下载 3 个 WAV 音频文件：")
    print("   - happy.wav: 内容「我今天很开心」，语气欢快")
    print("   - anxious.wav: 内容「我最近很焦虑」，语气低沉、颤抖")
    print("   - calm.wav: 内容「我感觉还可以」，语气平静")
    print("\n2. 音频要求：")
    print("   - 格式: WAV")
    print("   - 采样率: 16000 Hz")
    print("   - 声道: 单声道")
    print("   - 时长: 3-10 秒")
    print("\n3. 保存到目录: tests/test_audio/")
    print("\n【推荐工具】")
    print("  - Audacity (免费): https://www.audacityteam.org/")
    print("  - 在线转换: https://online-audio-converter.com/")
    print("\n【或使用在线 TTS】")
    print("  - 讯飞开放平台: https://www.xfyun.cn/services/online_tts")
    print("  - Azure TTS: https://azure.microsoft.com/zh-cn/services/cognitive-services/text-to-speech/")

print("\n" + "=" * 60)
print("准备完成后，运行: python scripts/manual/test_with_audio.py")
print("=" * 60)
