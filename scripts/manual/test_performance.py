import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
from src.tools.emotion import classify_emotion, classify_multimodal

print('=' * 60)
print('性能基准测试')
print('=' * 60)

test_texts = [
    "我今天很开心",
    "我最近总是失眠，很焦虑",
    "记不住事情了，很担心自己",
    "还好，没什么大问题",
]

# 测试1: 纯文本情绪识别延迟
print('\n【测试1】纯文本情绪识别延迟（规则方法）')
times_text = []
for text in test_texts:
    start = time.perf_counter()
    emotions = classify_emotion(text)
    elapsed = (time.perf_counter() - start) * 1000
    times_text.append(elapsed)
    print(f"  '{text[:20]}...' → {elapsed:.2f}ms")

avg_text = sum(times_text) / len(times_text)
print(f"\n  平均延迟: {avg_text:.2f}ms")

# 测试2: 模拟完整对话流程
print('\n【测试2】完整对话流程延迟估算')

stages = {
    'ASR (SenseVoice)': 70,
    '文本情绪识别': avg_text,
    '语音情绪识别': 150,  # Emotion2Vec+ 估算
    '多模态融合': 5,
    '记忆检索': 4,
    'LLM生成': 2000,
    'TTS合成': 500,
}

print('\n场景1: 默认配置（纯规则，无语音）')
total_default = 70 + avg_text + 4 + 2000 + 500
for stage, delay in stages.items():
    if stage not in ['语音情绪识别', '多模态融合']:
        percentage = delay / total_default * 100
        bar = '█' * int(percentage / 2)
        print(f'  {stage:<20} {delay:>7.1f}ms  {bar} {percentage:>5.1f}%')
print(f'  {"总计":<20} {total_default:>7.1f}ms')

print('\n场景2: 推荐配置（规则文本 + 语音情绪）')
total_recommended = sum(stages.values())
for stage, delay in stages.items():
    percentage = delay / total_recommended * 100
    bar = '█' * int(percentage / 2)
    print(f'  {stage:<20} {delay:>7.1f}ms  {bar} {percentage:>5.1f}%')
print(f'  {"总计":<20} {total_recommended:>7.1f}ms')

print('\n' + '=' * 60)
print('延迟对比总结')
print('=' * 60)
print(f'默认配置:     {total_default:.0f}ms   (baseline)')
print(f'推荐配置:     {total_recommended:.0f}ms   (+{total_recommended-total_default:.0f}ms, +{(total_recommended-total_default)/total_default*100:.1f}%)')
print(f'\n💡 结论:')
print(f'  - 情绪识别增加延迟: {total_recommended-total_default:.0f}ms')
print(f'  - 占总延迟比例: {(total_recommended-total_default)/total_recommended*100:.1f}%')
print(f'  - LLM仍是主要瓶颈: {2000/total_recommended*100:.1f}%')
print('=' * 60)
