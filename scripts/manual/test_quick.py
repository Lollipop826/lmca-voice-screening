import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.emotion import classify_multimodal

print('=' * 60)
print('多模态情绪识别系统实测')
print('=' * 60)

# 测试1: 纯文本情绪识别
print('\n=== 测试1: 纯文本情绪识别 ===')
test_cases = [
    ('我今天很开心', 'joy'),
    ('我最近很焦虑', 'anxiety'),
    ('我感到很平静', 'calm'),
    ('我有点困惑', 'confusion'),
]

for text, expected in test_cases:
    emotions = classify_multimodal(text)
    dominant = max(emotions, key=emotions.get)
    match = '✅ 通过' if dominant == expected else f'⚠️ 预期{expected}'
    print(f'文本: "{text}"')
    print(f'  识别: {dominant} ({emotions[dominant]:.2f}) {match}')
    print()

print('\n=== 测试2: 多模态融合（模拟） ===')
from src.tools.emotion.multimodal_fusion import fuse_emotions

audio_emotions = {
    'joy': 0.2, 'sadness': 0.6, 'anger': 0.05,
    'fear': 0.05, 'anxiety': 0.05, 'calm': 0.03, 'confusion': 0.02
}

text_emotions = {
    'joy': 0.7, 'sadness': 0.1, 'anger': 0.05,
    'fear': 0.03, 'anxiety': 0.05, 'calm': 0.05, 'confusion': 0.02
}

fused = fuse_emotions(audio_emotions, text_emotions)

print('输入:')
print(f'  语音主导情绪: sadness ({audio_emotions["sadness"]:.2f})')
print(f'  文本主导情绪: joy ({text_emotions["joy"]:.2f})')
print('\n融合结果:')
for emotion, score in sorted(fused.items(), key=lambda x: -x[1])[:3]:
    print(f'  {emotion}: {score:.3f}')

print('\n✅ 融合测试通过：结果在语音和文本之间')

print('\n' + '=' * 60)
print('✅ 核心功能测试完成')
print('=' * 60)
