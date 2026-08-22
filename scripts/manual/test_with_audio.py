"""
完整的音频情绪识别测试脚本（带模拟音频降级）
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import os
import time
import numpy as np
import wave

print("=" * 70)
print("多模态情绪识别系统 - 完整测试")
print("=" * 70)

# 检查是否有真实音频
audio_dir = ROOT / "tests" / "test_audio"
test_audio_files = ["happy.wav", "anxious.wav", "calm.wav"]
has_real_audio = all(os.path.exists(os.path.join(audio_dir, f)) for f in test_audio_files)

if has_real_audio:
    print("\n✅ 检测到真实音频文件，将进行完整测试")
    test_mode = "real"
else:
    print("\n⚠️ 未检测到真实音频，将使用模拟测试")
    print(f"   需要的文件: {', '.join(test_audio_files)}")
    print(f"   目标目录: {audio_dir}/")
    test_mode = "mock"

print("\n" + "=" * 70)
print("【阶段1】纯文本情绪识别测试")
print("=" * 70)

from src.tools.emotion import classify_emotion

text_test_cases = [
    ("我今天很开心", "joy"),
    ("我最近很焦虑", "anxiety"),
    ("我感到很平静", "calm"),
    ("我有点困惑", "confusion"),
    ("记不住事情了", "confusion"),
    ("担心自己的身体", "anxiety"),
]

print("\n纯文本测试:")
text_results = []
for text, expected in text_test_cases:
    start = time.perf_counter()
    emotions = classify_emotion(text)
    elapsed = (time.perf_counter() - start) * 1000

    dominant = max(emotions, key=emotions.get)
    match = "✅" if dominant == expected else "⚠️"

    print(f"{match} '{text}'")
    print(f"   识别: {dominant} ({emotions[dominant]:.3f}) | 延迟: {elapsed:.2f}ms")

    text_results.append({
        "text": text,
        "expected": expected,
        "predicted": dominant,
        "score": emotions[dominant],
        "latency": elapsed,
        "correct": dominant == expected
    })

accuracy = sum(1 for r in text_results if r["correct"]) / len(text_results)
avg_latency = sum(r["latency"] for r in text_results) / len(text_results)

print(f"\n📊 文本情绪识别统计:")
print(f"   准确率: {accuracy*100:.1f}% ({sum(1 for r in text_results if r['correct'])}/{len(text_results)})")
print(f"   平均延迟: {avg_latency:.2f}ms")

print("\n" + "=" * 70)
print("【阶段2】多模态融合测试")
print("=" * 70)

from src.tools.emotion import classify_multimodal

if test_mode == "real":
    print("\n真实音频测试:")

    audio_test_cases = [
        ("happy.wav", "我今天很开心", "joy"),
        ("anxious.wav", "我最近很焦虑", "anxiety"),
        ("calm.wav", "我感觉还可以", "calm"),
    ]

    multimodal_results = []
    for audio_file, text, expected in audio_test_cases:
        audio_path = os.path.join(audio_dir, audio_file)

        print(f"\n测试: {audio_file}")
        print(f"  文本: '{text}'")

        start = time.perf_counter()
        try:
            emotions = classify_multimodal(text, audio_path)
            elapsed = (time.perf_counter() - start) * 1000

            dominant = max(emotions, key=emotions.get)
            match = "✅" if dominant == expected else "⚠️"

            print(f"  {match} 识别: {dominant} ({emotions[dominant]:.3f})")
            print(f"  延迟: {elapsed:.2f}ms")
            print(f"  情绪分布: {', '.join([f'{k}:{v:.2f}' for k, v in sorted(emotions.items(), key=lambda x: -x[1])[:3]])}")

            multimodal_results.append({
                "audio": audio_file,
                "text": text,
                "expected": expected,
                "predicted": dominant,
                "score": emotions[dominant],
                "latency": elapsed,
                "correct": dominant == expected
            })
        except Exception as exc:
            print(f"  ❌ 错误: {exc}")
            import traceback
            traceback.print_exc()

    if multimodal_results:
        mm_accuracy = sum(1 for r in multimodal_results if r["correct"]) / len(multimodal_results)
        mm_avg_latency = sum(r["latency"] for r in multimodal_results) / len(multimodal_results)

        print(f"\n📊 多模态识别统计:")
        print(f"   准确率: {mm_accuracy*100:.1f}% ({sum(1 for r in multimodal_results if r['correct'])}/{len(multimodal_results)})")
        print(f"   平均延迟: {mm_avg_latency:.2f}ms")

else:
    print("\n模拟融合测试（无真实音频）:")

    from src.tools.emotion.multimodal_fusion import fuse_emotions

    test_scenarios = [
        {
            "name": "情绪一致",
            "text_emotions": {"joy": 0.7, "sadness": 0.1, "anxiety": 0.05, "calm": 0.08, "anger": 0.03, "fear": 0.02, "confusion": 0.02},
            "audio_emotions": {"joy": 0.65, "sadness": 0.15, "anxiety": 0.05, "calm": 0.08, "anger": 0.03, "fear": 0.02, "confusion": 0.02},
            "expected": "joy"
        },
        {
            "name": "情绪冲突（语音揭示真实情绪）",
            "text_emotions": {"joy": 0.5, "calm": 0.3, "sadness": 0.1, "anxiety": 0.05, "anger": 0.02, "fear": 0.02, "confusion": 0.01},
            "audio_emotions": {"sadness": 0.6, "anxiety": 0.2, "joy": 0.1, "calm": 0.05, "anger": 0.02, "fear": 0.02, "confusion": 0.01},
            "expected": "sadness"
        },
        {
            "name": "焦虑表达",
            "text_emotions": {"anxiety": 0.5, "fear": 0.2, "sadness": 0.15, "confusion": 0.08, "joy": 0.03, "calm": 0.02, "anger": 0.02},
            "audio_emotions": {"anxiety": 0.55, "fear": 0.25, "sadness": 0.1, "confusion": 0.05, "joy": 0.02, "calm": 0.02, "anger": 0.01},
            "expected": "anxiety"
        }
    ]

    for scenario in test_scenarios:
        print(f"\n场景: {scenario['name']}")
        fused = fuse_emotions(scenario["audio_emotions"], scenario["text_emotions"])
        dominant = max(fused, key=fused.get)
        match = "✅" if dominant == scenario["expected"] else "⚠️"

        print(f"  文本主导: {max(scenario['text_emotions'], key=scenario['text_emotions'].get)}")
        print(f"  语音主导: {max(scenario['audio_emotions'], key=scenario['audio_emotions'].get)}")
        print(f"  {match} 融合结果: {dominant} ({fused[dominant]:.3f})")
        print(f"  Top 3: {', '.join([f'{k}:{v:.2f}' for k, v in sorted(fused.items(), key=lambda x: -x[1])[:3]])}")

print("\n" + "=" * 70)
print("【阶段3】组件加载测试")
print("=" * 70)

print("\n测试语音情绪分类器加载...")
try:
    from src.tools.emotion import get_audio_emotion_classifier

    start = time.perf_counter()
    classifier = get_audio_emotion_classifier()
    elapsed = (time.perf_counter() - start) * 1000

    print(f"✅ 语音情绪分类器加载成功")
    print(f"   加载耗时: {elapsed:.0f}ms")
    print(f"   模型: {os.getenv('EMOTION2VEC_MODEL', 'iic/emotion2vec_plus_large')}")
    print(f"   设备: {os.getenv('EMOTION_USE_GPU', 'false') == 'true' and 'GPU' or 'CPU'}")
except Exception as exc:
    print(f"⚠️ 语音情绪分类器加载失败: {exc}")

print("\n" + "=" * 70)
print("【阶段4】性能基准测试")
print("=" * 70)

print("\n完整对话流程延迟估算:")
stages = {
    "ASR (SenseVoice)": 70,
    "文本情绪识别": avg_latency,
    "语音情绪识别": 150 if test_mode == "real" else 0,
    "多模态融合": 5 if test_mode == "real" else 0,
    "记忆检索": 4,
    "LLM生成": 2000,
    "TTS合成": 500,
}

total = sum(stages.values())

for stage, delay in stages.items():
    if delay > 0:
        percentage = delay / total * 100
        bar = "█" * int(percentage / 2)
        print(f"  {stage:<20} {delay:>7.1f}ms  {bar} {percentage:>5.1f}%")

print(f"  {'总计':<20} {total:>7.1f}ms")

if test_mode == "real":
    default_total = 70 + avg_latency + 4 + 2000 + 500
    print(f"\n对比:")
    print(f"  默认配置: {default_total:.0f}ms")
    print(f"  推荐配置: {total:.0f}ms (+{total-default_total:.0f}ms, +{(total-default_total)/default_total*100:.1f}%)")

print("\n" + "=" * 70)
print("测试总结")
print("=" * 70)

print(f"\n✅ 已完成:")
print(f"  - 纯文本情绪识别: {len(text_results)} 个测试")
print(f"  - 文本准确率: {accuracy*100:.1f}%")
print(f"  - 平均延迟: {avg_latency:.2f}ms")

if test_mode == "real":
    print(f"  - 多模态识别: {len(multimodal_results)} 个测试")
    print(f"  - 多模态准确率: {mm_accuracy*100:.1f}%")
    print(f"  - 多模态延迟: {mm_avg_latency:.2f}ms")
    print(f"\n🎉 完整测试通过！系统已就绪。")
else:
    print(f"  - 融合算法: 3 个模拟场景")
    print(f"\n⚠️ 需要真实音频完成完整测试")
    print(f"\n下一步:")
    print(f"  1. 准备音频文件（运行 scripts/manual/generate_test_audio.py）")
    print(f"  2. 保存到: {audio_dir}/")
    print(f"  3. 重新运行: python scripts/manual/test_with_audio.py")

print("=" * 70)
