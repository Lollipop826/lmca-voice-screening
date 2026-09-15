#!/usr/bin/env python3
"""Run the frozen high-discrimination 2x2 response-quality study.

This is a thin, reproducible wrapper around the already verified realtime
benchmark harness.  It changes only the frozen stimulus list; the application
service, event protocol, integrity checks, and arm randomization are reused.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import benchmark_realtime_study as study


# 14 samples: four memory-dependent prompts, two memory traps, three
# same-text/different-prosody pairs (six samples), and two neutral controls.
# For each paired block the exact prepared WAV is reused across all four arms.
study.CASES = [
    # Memory-dependent (4)
    ('mem-family-followup', '高辨别力·记忆依赖·家庭',
     '我这两天还是因为女儿不回消息觉得难受。', 'sad',
     '衔接已知的家庭沟通背景；不编造女儿想法或冲突细节。'),
    ('mem-sleep-followup', '高辨别力·记忆依赖·睡眠',
     '最近又睡不好，晚上总忍不住看工作消息。', 'neutral',
     '可衔接已同意的睡前手机放远习惯；不武断诊断。'),
    ('mem-method-followup', '高辨别力·记忆依赖·指代',
     '上次说的睡前那个办法我试了两天，还是拿不准下一步。', 'confusion',
     '正确解析为睡前手机放远的方法；不能编造其他办法。'),
    ('mem-family-update', '高辨别力·记忆依赖·更新覆盖',
     '我和女儿已经把误会说开了，今天想分享她工作上的好消息。', 'happy',
     '以已经和好这一新事实为准；不要继续把未回消息当现状。'),

    # Memory traps (2)
    ('trap-walk-update', '高辨别力·记忆陷阱·事实纠正',
     '散步现在改到周五下午了，周三要上课，别再按旧时间记了。', 'neutral',
     '认可新时间；不能继续强调周三散步；关闭写入时不虚称已永久保存。'),
    ('trap-keys-unrelated', '高辨别力·记忆陷阱·无关',
     '今天买菜回来发现钥匙忘在家里了，在门口等了半天。', 'neutral',
     '回应眼前经历；不强扯家庭、睡眠或散步历史。'),

    # Same text, different prosody (3 pairs; six samples)
    ('prosody-achievement-happy', '高辨别力·情绪成对·完成-积极',
     '这件事终于完成了，我现在想和你说说我的感受。', 'happy',
     '识别并匹配积极/释然语气；回应具体感受，不过度夸大。'),
    ('prosody-achievement-sad', '高辨别力·情绪成对·完成-低落',
     '这件事终于完成了，我现在想和你说说我的感受。', 'sad',
     '识别并匹配低落语气；先倾听，不把完成自动解释成开心。'),
    ('prosody-boundary-anger', '高辨别力·情绪成对·边界-生气',
     '我希望这次能按照我说的方式来处理。', 'anger',
     '识别边界和生气语气；尊重边界，不激化冲突。'),
    ('prosody-boundary-neutral', '高辨别力·情绪成对·边界-平静',
     '我希望这次能按照我说的方式来处理。', 'neutral',
     '保持平实回应；不凭空添加愤怒或委屈。'),
    ('prosody-plan-fear', '高辨别力·情绪成对·办事-紧张',
     '明天我需要一个人去办理这件事。', 'fear',
     '接住紧张但不过度诊断；给出轻量、可执行的支持。'),
    ('prosody-plan-neutral', '高辨别力·情绪成对·办事-平静',
     '明天我需要一个人去办理这件事。', 'neutral',
     '保持平实；不把普通计划误读为焦虑。'),

    # Neutral controls (2)
    ('control-cooking', '高辨别力·中性控制·实用',
     '冰箱里还有两个鸡蛋和一根黄瓜，能做点什么简单的？', 'neutral',
     '给出与食材匹配的简短可执行建议；不做心理分析。'),
    ('control-desk', '高辨别力·中性控制·日常',
     '今天下午有空的话，我想整理一下书桌。', 'neutral',
     '给出自然、轻量的回应；不强行引用历史或情绪。'),
]


if __name__ == '__main__':
    study.main()
