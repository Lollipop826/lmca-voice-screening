"""
情绪识别模型性能测试
测试不同配置对响应速度的影响
"""
import time
import os
from typing import Dict, List

# 测试配置
TEST_TEXTS = [
    "我今天很开心",
    "我最近总是失眠，很焦虑",
    "记不住事情了，很担心自己",
    "还好，没什么大问题",
]


def benchmark_text_emotion():
    """测试文本情绪识别延迟"""
    print("=" * 60)
    print("文本情绪识别性能测试")
    print("=" * 60)

    results = {}
    remote_models = os.getenv("EMOTION_BENCHMARK_REMOTE", "false").lower() in {
        "1", "true", "yes", "on"
    }

    # =================== 测试1: 纯规则（默认配置） ===================
    print("\n【测试1】纯规则方法（默认）")
    os.environ["EMOTION_USE_TRANSFORMERS"] = "false"

    from src.tools.emotion import EmotionClassifier
    classifier_rule = EmotionClassifier(use_transformers=False)

    times_rule = []
    for text in TEST_TEXTS:
        start = time.perf_counter()
        emotions = classifier_rule.classify(text)
        elapsed = (time.perf_counter() - start) * 1000
        times_rule.append(elapsed)
        print(f"  '{text[:15]}...' → {elapsed:.2f}ms")

    avg_rule = sum(times_rule) / len(times_rule)
    results["纯规则"] = avg_rule
    print(f"  平均延迟: {avg_rule:.2f}ms")

    # =================== 测试2: RoBERTa 点评模型 ===================
    print("\n【测试2】RoBERTa + 点评模型（现有配置）")

    try:
        classifier_dianping = EmotionClassifier(
            model_name="uer/roberta-base-finetuned-dianping-chinese",
            use_transformers=True,
            local_files_only=not remote_models,
        )

        # 预热（首次加载模型）
        print("  正在加载模型...")
        warmup_start = time.perf_counter()
        classifier_dianping.classify("测试")
        warmup_time = (time.perf_counter() - warmup_start) * 1000
        print(f"  模型加载时间: {warmup_time:.0f}ms（仅首次）")

        # 实际测试
        times_dianping = []
        for text in TEST_TEXTS:
            start = time.perf_counter()
            emotions = classifier_dianping.classify(text)
            elapsed = (time.perf_counter() - start) * 1000
            times_dianping.append(elapsed)
            print(f"  '{text[:15]}...' → {elapsed:.2f}ms")

        avg_dianping = sum(times_dianping) / len(times_dianping)
        results["RoBERTa点评"] = avg_dianping
        print(f"  平均延迟: {avg_dianping:.2f}ms")

    except Exception as e:
        print(f"  ⚠️ 跳过（需安装transformers）: {e}")
        results["RoBERTa点评"] = None

    # =================== 测试3: GoEmotions 模型 ===================
    print("\n【测试3】GoEmotions-zh（推荐模型）")

    try:
        classifier_goemotions = EmotionClassifier(
            model_name="SamLowe/roberta-base-go_emotions-chinese",
            use_transformers=True,
            local_files_only=not remote_models,
        )

        # 预热
        print("  正在加载模型...")
        warmup_start = time.perf_counter()
        classifier_goemotions.classify("测试")
        warmup_time = (time.perf_counter() - warmup_start) * 1000
        print(f"  模型加载时间: {warmup_time:.0f}ms（仅首次）")

        # 实际测试
        times_goemotions = []
        for text in TEST_TEXTS:
            start = time.perf_counter()
            emotions = classifier_goemotions.classify(text)
            elapsed = (time.perf_counter() - start) * 1000
            times_goemotions.append(elapsed)
            print(f"  '{text[:15]}...' → {elapsed:.2f}ms")

        avg_goemotions = sum(times_goemotions) / len(times_goemotions)
        results["GoEmotions"] = avg_goemotions
        print(f"  平均延迟: {avg_goemotions:.2f}ms")

    except Exception as e:
        print(f"  ⚠️ 跳过（需联网下载）: {e}")
        results["GoEmotions"] = None

    # =================== 汇总 ===================
    print("\n" + "=" * 60)
    print("汇总对比")
    print("=" * 60)
    print(f"{'方法':<20} {'平均延迟':<15} {'相对慢倍数'}")
    print("-" * 60)

    baseline = results["纯规则"]
    for method, avg_time in results.items():
        if avg_time is not None:
            relative = avg_time / baseline
            print(f"{method:<20} {avg_time:>8.2f}ms      {relative:>5.1f}x")
        else:
            print(f"{method:<20} {'N/A':<15} {'N/A'}")

    return results


def benchmark_audio_emotion():
    """测试语音情绪识别延迟"""
    print("\n" + "=" * 60)
    print("语音情绪识别性能测试")
    print("=" * 60)

    try:
        from src.tools.emotion import get_audio_emotion_classifier

        # 检查是否有测试音频
        import os
        test_audio = "tests/test_audio/sample.wav"
        if not os.path.exists(test_audio):
            print("⚠️ 未找到测试音频，跳过")
            print("提示：创建 tests/test_audio/sample.wav 以测试")
            return

        classifier = get_audio_emotion_classifier()

        # 预热
        print("正在加载 Emotion2Vec+ 模型...")
        warmup_start = time.perf_counter()
        classifier.classify_audio(test_audio)
        warmup_time = (time.perf_counter() - warmup_start) * 1000
        print(f"模型加载时间: {warmup_time:.0f}ms（仅首次）")

        # 实际测试
        times = []
        for i in range(5):
            start = time.perf_counter()
            emotions = classifier.classify_audio(test_audio)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            print(f"  第{i+1}次: {elapsed:.2f}ms")

        avg = sum(times) / len(times)
        print(f"\n平均延迟: {avg:.2f}ms")

    except Exception as e:
        print(f"⚠️ 语音测试失败: {e}")


def benchmark_full_pipeline():
    """测试完整对话流程延迟"""
    print("\n" + "=" * 60)
    print("完整对话流程延迟测试")
    print("=" * 60)

    # 模拟完整流程
    stages = {
        "ASR (SenseVoice)": 70,  # 已知数据
        "文本情绪识别": None,  # 待测
        "语音情绪识别": None,  # 待测
        "多模态融合": 5,  # 估计
        "记忆检索": 4,  # 已知数据
        "LLM生成": 2000,  # 假设
        "TTS合成": 500,  # 假设
    }

    print("\n【场景1】纯规则（当前默认）")
    stages["文本情绪识别"] = 2  # 规则方法
    stages["语音情绪识别"] = 0  # 不启用
    total_rule = sum(v for v in stages.values() if v is not None)
    print_pipeline(stages, total_rule)

    print("\n【场景2】规则 + Emotion2Vec+（推荐配置）")
    stages["文本情绪识别"] = 2  # 保持规则
    stages["语音情绪识别"] = 150  # Emotion2Vec+
    total_audio = sum(v for v in stages.values() if v is not None)
    print_pipeline(stages, total_audio)

    print("\n【场景3】RoBERTa + Emotion2Vec+（全模型）")
    stages["文本情绪识别"] = 50  # RoBERTa
    stages["语音情绪识别"] = 150  # Emotion2Vec+
    total_full = sum(v for v in stages.values() if v is not None)
    print_pipeline(stages, total_full)

    print("\n" + "=" * 60)
    print("延迟对比")
    print("=" * 60)
    print(f"{'配置':<30} {'总延迟':<15} {'增加延迟'}")
    print("-" * 60)
    print(f"{'场景1: 纯规则（默认）':<30} {total_rule:>8}ms      baseline")
    print(f"{'场景2: 规则+语音（推荐）':<30} {total_audio:>8}ms      +{total_audio-total_rule}ms")
    print(f"{'场景3: 全模型':<30} {total_full:>8}ms      +{total_full-total_rule}ms")

    print("\n💡 结论:")
    print(f"  - 推荐配置（场景2）增加 {total_audio-total_rule}ms，占总延迟的 {(total_audio-total_rule)/total_audio*100:.1f}%")
    print(f"  - 主要延迟在 LLM 和 TTS，情绪识别影响很小")


def print_pipeline(stages: Dict[str, float], total: float):
    """打印流程各阶段延迟"""
    for stage, time_ms in stages.items():
        if time_ms is not None and time_ms > 0:
            percentage = time_ms / total * 100
            bar = "█" * int(percentage / 2)
            print(f"  {stage:<20} {time_ms:>6}ms  {bar} {percentage:>5.1f}%")
    print(f"  {'总计':<20} {total:>6}ms")


if __name__ == "__main__":
    benchmark_text_emotion()
    benchmark_audio_emotion()
    benchmark_full_pipeline()

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
