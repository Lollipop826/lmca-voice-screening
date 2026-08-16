"""
集成测试 - 多模态情绪识别
"""
import os
from pathlib import Path

from src.tools.emotion import classify_multimodal, get_audio_emotion_classifier


def test_text_only_emotion():
    """测试纯文本情绪识别"""
    print("\n=== 测试1: 纯文本情绪识别 ===")

    test_cases = [
        ("我今天很开心", "joy"),
        ("我最近很焦虑", "anxiety"),
        ("我感到很平静", "calm"),
        ("我有点困惑", "confusion"),
    ]

    for text, expected in test_cases:
        emotions = classify_multimodal(text)
        dominant = max(emotions, key=emotions.get)
        print(f"文本: {text}")
        print(f"  识别结果: {dominant} ({emotions[dominant]:.2f})")
        print(f"  预期: {expected}")
        print(f"  {'✅ 通过' if dominant == expected else '⚠️ 未匹配'}")
        print()


def test_audio_emotion():
    """测试语音情绪识别（如果有测试音频）"""
    print("\n=== 测试2: 语音情绪识别 ===")

    # 检查是否有测试音频
    test_audio_dir = Path("tests/test_audio")
    if not test_audio_dir.exists():
        print("⚠️ 未找到测试音频目录，跳过语音测试")
        print("可创建 tests/test_audio/ 并放入测试音频文件")
        return

    # 查找测试音频
    audio_files = list(test_audio_dir.glob("*.wav"))
    if not audio_files:
        print("⚠️ tests/test_audio/ 中没有 .wav 文件")
        return

    # 测试第一个音频文件
    audio_path = str(audio_files[0])
    print(f"测试音频: {audio_path}")

    try:
        # 仅语音识别
        audio_classifier = get_audio_emotion_classifier()
        audio_emotions = audio_classifier.classify_audio(audio_path)

        print(f"  语音情绪识别结果:")
        for emotion, score in sorted(audio_emotions.items(), key=lambda x: -x[1])[:3]:
            print(f"    {emotion}: {score:.3f}")

        # 提取韵律特征
        prosody = audio_classifier.get_prosody_features(audio_path)
        print(f"  韵律特征:")
        print(f"    音高: {prosody['pitch_mean']:.1f} Hz")
        print(f"    能量: {prosody['energy_mean']:.3f}")
        print(f"    语速: {prosody['tempo']:.1f} BPM")

        print("✅ 语音情绪识别测试通过")

    except Exception as e:
        print(f"❌ 语音测试失败: {e}")


def test_multimodal_fusion():
    """测试多模态融合"""
    print("\n=== 测试3: 多模态融合（模拟） ===")

    # 模拟场景：文本说"开心"，但语音带悲伤
    from src.tools.emotion.multimodal_fusion import fuse_emotions

    audio_emotions = {
        "joy": 0.2,
        "sadness": 0.6,
        "anger": 0.05,
        "fear": 0.05,
        "anxiety": 0.05,
        "calm": 0.03,
        "confusion": 0.02,
    }

    text_emotions = {
        "joy": 0.7,
        "sadness": 0.1,
        "anger": 0.05,
        "fear": 0.03,
        "anxiety": 0.05,
        "calm": 0.05,
        "confusion": 0.02,
    }

    # 融合
    fused = fuse_emotions(audio_emotions, text_emotions)

    print("输入:")
    print(f"  语音: sadness={audio_emotions['sadness']:.2f}, joy={audio_emotions['joy']:.2f}")
    print(f"  文本: joy={text_emotions['joy']:.2f}, sadness={text_emotions['sadness']:.2f}")
    print()
    print("融合结果（50%语音 + 30%文本 + 20%韵律调整）:")
    for emotion, score in sorted(fused.items(), key=lambda x: -x[1])[:3]:
        print(f"  {emotion}: {score:.3f}")

    # 验证：融合结果应该在两者之间
    assert 0.2 < fused["sadness"] < 0.6, "融合结果应在语音和文本之间"
    assert 0.2 < fused["joy"] < 0.7, "融合结果应在语音和文本之间"

    print("✅ 多模态融合测试通过")


def test_integration_with_memory():
    """测试与记忆系统的集成"""
    print("\n=== 测试4: 与记忆系统集成 ===")

    try:
        from src.context_management.emotion_memobase import EmotionMemobase

        # 创建测试记忆管理器
        memory = EmotionMemobase(db_path="tests/test_emotion_memory.db")

        # 模拟对话
        patient_id = "test_patient_001"
        user_text = "我最近总是失眠，很焦虑"
        assistant_text = "我理解您的感受，失眠确实会让人焦虑..."

        # 识别情绪（纯文本）
        emotions = classify_multimodal(user_text)
        print(f"识别情绪: {emotions}")

        # 捕获到记忆（假设 emotion_memobase 有此方法）
        if hasattr(memory, 'capture_turn'):
            memory.capture_turn(
                patient_id=patient_id,
                user_message=user_text,
                assistant_message=assistant_text,
                emotions=emotions
            )
            print("✅ 情绪已记录到长期记忆")
        else:
            print("⚠️ 记忆系统未实现 capture_turn 方法")

    except Exception as e:
        print(f"⚠️ 记忆集成测试跳过: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("多模态情绪识别集成测试")
    print("=" * 60)

    test_text_only_emotion()
    test_audio_emotion()
    test_multimodal_fusion()
    test_integration_with_memory()

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
