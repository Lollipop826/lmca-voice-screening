#!/usr/bin/env python3
"""Offline report of audited realtime trials, including explicitly terminated studies.

Never makes model/network calls. The original all-record summary is preserved;
the balanced complete-block analysis is written to a separate JSON artifact.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_voice_realtime_ablation import audit
from scripts.benchmark_voice_realtime_ablation import ARMS, METRIC_PAIRS, numeric_summary, sha, summary

LABELS = {'M1E1': '记忆＋情绪均开启', 'M0E1': '关闭记忆', 'M1E0': '关闭情绪', 'M0E0': '记忆、情绪均关闭'}
SAMPLE_LABELS = {
    'memory-family-conflict-followup': '家庭联系',
    'memory-sleep-followup': '睡眠与工作消息',
    'memory-community-walk-followup': '日常散步',
    'memory-unrelated-control': '忘带钥匙（无关对照）',
}
COMPONENTS = [
    ('speech_end_to_vad_end_ms', '语音末帧→VAD结束'),
    ('vad_end_to_asr_final_ms', 'VAD结束→最终转写'),
    ('asr_final_to_first_ai_text_ms', '最终转写→首段AI文本'),
    ('first_ai_text_to_first_audio_ms', '首段AI文本→首非空音频'),
]
DISPOSITIONS = {
    'valid': '完整成功', 'llm_quota_failed': 'LLM配额拒绝',
    'cancelled_before_measured_input': '准备阶段主动中止', 'other_failure': '其他失败',
}
COMPARISONS = [('M1E1', 'M0E1'), ('M1E0', 'M0E0'), ('M1E1', 'M1E0'), ('M0E1', 'M0E0'), ('M1E1', 'M0E0')]


def f(value, decimals=2):
    return f'{value:.{decimals}f}' if isinstance(value, (int, float)) and math.isfinite(value) else '—'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def md_rows(rows):
    return '\n'.join('| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows)


def complete_block_summary(protocol, records):
    """Use every all-four-success block, not arm-wise or latency-based selection."""
    schedule = {s['trial_id']: s for s in protocol['schedule']}
    if len(schedule) != len(protocol['schedule']):
        raise ValueError('Duplicate planned trial ID')
    by_id = {}
    for r in records:
        tid = r['trial_id']
        if tid in by_id:
            raise ValueError('Duplicate observed trial ID: ' + tid)
        if tid not in schedule or any(r.get(k) != v for k, v in schedule[tid].items()):
            raise ValueError('Record does not match frozen schedule: ' + tid)
        by_id[tid] = r
    blocks = defaultdict(list)
    for spec in protocol['schedule']:
        blocks[spec['block_id']].append(spec)
    complete, block_rows = [], []
    for bid, specs in sorted(blocks.items()):
        if len(specs) != len(ARMS) or {s['arm'] for s in specs} != set(ARMS):
            raise ValueError('Block must contain each arm exactly once')
        if len({(s['sample_id'], s['repeat']) for s in specs}) != 1:
            raise ValueError('Within-block stimulus/repeat mismatch')
        valid_ids = [s['trial_id'] for s in specs if s['trial_id'] in by_id and by_id[s['trial_id']]['valid_for_latency']]
        is_complete = len(valid_ids) == len(ARMS)
        if is_complete:
            complete.append(bid)
        block_rows.append({'block_id': bid, 'sample_id': specs[0]['sample_id'], 'repeat': specs[0]['repeat'],
                           'all_four_valid': is_complete, 'valid_trial_ids': valid_ids})
    main = [r for r in records if r['block_id'] in complete]
    extra = [r for r in records if r['valid_for_latency'] and r['block_id'] not in complete]
    groups = {}
    for arm in ARMS:
        rows = [r for r in main if r['arm'] == arm]
        groups[arm] = {
            'n': len(rows), 'trial_ids': [r['trial_id'] for r in rows],
            'metrics': {k: numeric_summary(r['metrics'].get(k) for r in rows) for k in METRIC_PAIRS},
            'output_text_chars': numeric_summary(r.get('output_text_chars') for r in rows),
            'output_audio_duration_s': numeric_summary(r.get('output_audio_duration_s') for r in rows),
        }
    paired = {}
    block_map = {bid: {r['arm']: r for r in main if r['block_id'] == bid} for bid in complete}
    for left, right in COMPARISONS:
        pair_rows = []
        for bid, arms in block_map.items():
            pair_rows.append({'block_id': bid, 'metrics': {
                key: arms[left]['metrics'][key] - arms[right]['metrics'][key]
                for key in METRIC_PAIRS if key in arms[left]['metrics'] and key in arms[right]['metrics']}})
        paired[left + ' - ' + right] = {'n': len(pair_rows), 'rows': pair_rows,
            'metrics': {key: numeric_summary(r['metrics'].get(key) for r in pair_rows) for key in METRIC_PAIRS}}
    return {
        'schema_version': 'realtime-complete-block-analysis-v1',
        'analysis_rule': 'Retain all and only blocks with all four valid arms; never select on latency. The incomplete-study reporting rule was added after quota interruption; the measurement protocol was not changed.',
        'analysis_scope': 'Exploratory small-sample complete-block comparison, not a completed 48-trial study',
        'planned_formal': len(schedule), 'attempted_formal': len(records),
        'valid_formal': sum(r['valid_for_latency'] for r in records),
        'missing_trial_ids': [tid for tid in schedule if tid not in by_id],
        'complete_block_ids': complete, 'main_trial_ids': [r['trial_id'] for r in main],
        'main_n': len(main), 'n_per_arm': len(complete),
        'main_sample_block_counts': dict(Counter(b['sample_id'] for b in block_rows if b['all_four_valid'])),
        'valid_outside_complete_blocks': [r['trial_id'] for r in extra],
        'blocks': block_rows, 'groups': groups, 'paired_deltas': paired,
    }


def make(out: Path, canvas_path: Path):
    out, canvas_path = out.resolve(), canvas_path.resolve()
    all_summary = summary(out)  # Local files only; retains ALL formal records and original pairing.
    audited = audit(out)
    if not audited['passed'] or not (audited['complete_schedule'] or audited['study_terminated_early']):
        raise RuntimeError('Report requires passing audit and a complete or explicitly terminated run')
    protocol = json.loads((out / 'realtime_protocol.json').read_text())
    records = [json.loads(p.read_text()) for p in sorted((out / 'realtime_trials').glob('formal-*/result.json'))]
    balanced = complete_block_summary(protocol, records)
    if not balanced['complete_block_ids']:
        raise RuntimeError('No complete four-arm block: cannot publish a comparative latency table')
    by_audit = {r['trial_id']: r for r in audited['rows']}
    counts = audited['disposition_counts']
    accounting = {'planned': len(protocol['schedule']), 'started': len(records),
                  'successful': counts['valid'], 'llm_quota_failed': counts['llm_quota_failed'],
                  'cancelled_before_measured_input': counts['cancelled_before_measured_input'],
                  'other_failed': counts['other_failure'], 'not_run': len(balanced['missing_trial_ids'])}
    assert sum(accounting[k] for k in ['successful', 'llm_quota_failed', 'cancelled_before_measured_input', 'other_failed', 'not_run']) == accounting['planned']
    assert all_summary['valid_formal'] == accounting['successful']
    balanced.update({'created_at': datetime.now().astimezone().isoformat(),
                     'study_terminated_early': audited['study_terminated_early'],
                     'complete_schedule': audited['complete_schedule'], 'formal_accounting': accounting,
                     'protocol_sha256': sha(out / 'realtime_protocol.json'),
                     'all_record_summary': str(out / 'realtime_summary.json')})
    save(out / 'complete_block_summary.json', balanced)
    main_ids = set(balanced['main_trial_ids'])
    main = [r for r in records if r['trial_id'] in main_ids]
    good = [r for r in records if r['valid_for_latency']]
    groups = balanced['groups']
    full = groups['M1E1']['metrics']
    e2e = full['speech_end_to_first_audio_ms']
    first_text = full['speech_end_to_first_ai_text_ms']
    n = balanced['n_per_arm']
    start, end = min(r['started_at'] for r in records), max(r['finished_at'] for r in records)
    main_start, main_end = min(r['started_at'] for r in main), max(r['finished_at'] for r in main)
    enabled = [r for r in main if ARMS[r['arm']]['emotion']]
    disabled = [r for r in main if not ARMS[r['arm']]['emotion']]
    emotion_used = sum(r['final_insight']['emotion'].get('audio_model_used') is True for r in enabled)
    emotion_disabled = sum(r['final_insight']['emotion'].get('source') == 'disabled' for r in disabled)
    emotion_ms = numeric_summary(r['final_insight']['emotion'].get('inference_ms') for r in enabled)
    pace_max = max(r['pacing']['speech_frame_lag_ms'].get('max', 0) for r in records)
    card_ms = numeric_summary(by_audit[r['trial_id']]['memory_card_setup_ms'] for r in main)
    arm_accounting = []
    for arm in ARMS:
        specs = [s for s in protocol['schedule'] if s['arm'] == arm]
        rows = [r for r in records if r['arm'] == arm]
        kinds = Counter(by_audit[r['trial_id']]['disposition'] for r in rows)
        arm_accounting.append({'arm': arm, 'label': LABELS[arm], 'planned': len(specs), 'started': len(rows),
            'valid': kinds['valid'], 'quota': kinds['llm_quota_failed'], 'cancelled': kinds['cancelled_before_measured_input'],
            'other': kinds['other_failure'], 'not_run': len(specs) - len(rows), 'main': groups[arm]['n']})
    save(out / 'formal_accounting.json', {'totals': accounting, 'groups': arm_accounting,
        'trials': [{'trial_id': r['trial_id'], 'arm': r['arm'], 'block_id': r['block_id'],
                    'disposition': by_audit[r['trial_id']]['disposition'], 'included_in_main_analysis': r['trial_id'] in main_ids,
                    'quota_request_ids': by_audit[r['trial_id']]['llm_quota_request_ids']} for r in records],
        'not_run_trial_ids': balanced['missing_trial_ids']})
    condition_rows, component_rows, completion_rows, all_rows = [], [], [], []
    for arm, g in groups.items():
        m, label = g['metrics'], LABELS[arm]
        e = m['speech_end_to_first_audio_ms']
        condition_rows.append([label, g['n'], f(m['speech_end_to_first_ai_text_ms']['mean']), f(e['mean']),
                               f(e['median']), f(e['max']), f(e['std'])])
        component_rows.append([label] + [f(m[k]['mean']) for k, _ in COMPONENTS])
        completion_rows.append([label, f(m['first_ai_text_to_tts_end_ms']['mean']),
                                f(g['output_text_chars']['mean']), f(g['output_audio_duration_s']['mean'])])
        ag = all_summary['groups'][arm]
        all_rows.append([label, ag['valid'], f(ag['metrics']['speech_end_to_first_ai_text_ms']['mean']),
                         f(ag['metrics']['speech_end_to_first_audio_ms']['mean'])])
    paired_rows = []
    for comp, v in balanced['paired_deltas'].items():
        left, right = comp.split(' - ')
        m = v['metrics']['speech_end_to_first_audio_ms']
        paired_rows.append([LABELS[left] + ' − ' + LABELS[right], v['n'], f(m['mean']), f(m['median']), f(m['min']), f(m['max'])])
    material_rows = [[SAMPLE_LABELS[s['sample_id']], s['reference_text'], f(s['speech_seconds'], 3),
                      balanced['main_sample_block_counts'].get(s['sample_id'], 0)] for s in protocol['samples']]
    extras = ', '.join(balanced['valid_outside_complete_blocks']) or '无'
    report_path = ROOT / 'docs' / f'realtime_voice_ablation_report_{out.name.removeprefix("real_voice_ablation_")}.md'
    status_title = '部分完成：LLM 免费额度耗尽后提前停止' if audited['study_terminated_early'] else '计划轮次已全部执行'
    report = f'''# 实时语音消融实验复测报告

> **实验状态：{status_title}。**
> **部署范围：本地 Silero VAD＋火山流式 ASR＋SQLite 记忆卡＋Emotion2Vec＋DashScope LLM＋火山 TTS。**
> 不是 SoulX＋Memobase 完整部署成绩；不是扬声器开始播放的实测。

## 1. 结论与完成情况

- 正式运行：**{start} 至 {end}**（北京时间）。
- 计划 **{accounting['planned']} 轮**，实际启动 **{accounting['started']} 轮**：完整成功 **{accounting['successful']} 轮**，LLM 配额拒绝 **{accounting['llm_quota_failed']} 轮**，准备阶段主动中止 **{accounting['cancelled_before_measured_input']} 轮**，未执行 **{accounting['not_run']} 轮**。未用历史数据或追加成功样本填补失败。
- 主消融比较采用 **{n} 个四组均成功的完整区组，共 {balanced['main_n']} 轮，每组 {n} 轮**；额外 {len(balanced['valid_outside_complete_blocks'])} 轮成功仅放入全量描述，避免四组素材构成不一致。
- 主分析两模块均开启时：语音末帧→首段 AI 文本平均 **{f(first_text['mean'])} ms**；→客户端首非空回复音频平均 **{f(e2e['mean'])} ms（{f(e2e['mean']/1000)} s）**，中位数 **{f(e2e['median'])} ms**。
- 独立从原始收发日志复算 **{audited['recomputed_metric_count']} 个时间差，全部一致**。审计通过只表示记录可信，不表示原计划已完成。

### 1.1 为什么没有完成原计划

`formal-023` 起，DashScope 的 `qwen3.7-flash` 返回 **HTTP 403 / `AllocationQuota.FreeTierOnly` / `Free quota exhausted`**。日志中的 {audited['unique_llm_quota_requests']} 个唯一 request_id 已按会话关联并核验，对应 `formal-023`～`formal-046`。不是火山 ASR/TTS 返回的停用错误：这 {accounting['llm_quota_failed']} 轮已收到最终转写，但未能生成正常回复音频。

检测并确认持续配额拒绝后停止请求。`formal-047` 在开场准备阶段被主动中止，测量输入帧数为 0；`formal-048` 未启动。**没有切换模型，没有开启付费，也没有后台继续重试。** 该执行器未在第一次配额错误时自动熔断，这也是后续需补的实验运行保护。

| 配置 | 计划 | 启动 | 完整成功 | 配额拒绝 | 准备期中止 | 未执行 | 主分析 |
|---|---:|---:|---:|---:|---:|---:|---:|
{md_rows([[g['label'],g['planned'],g['started'],g['valid'],g['quota'],g['cancelled'],g['not_run'],g['main']] for g in arm_accounting])}

## 2. 设计、控制条件与实际部署

### 2.1 四组与配对方式

| 组别 | SQLite 长期记忆卡 | Emotion2Vec 音频＋文本情绪 |
|---|---|---|
| M1E1：均开启 | 开 | 开 |
| M0E1：关闭记忆 | 关，保留相同基础档案 | 开 |
| M1E0：关闭情绪 | 开 | 关 |
| M0E0：双关闭 | 关，保留相同基础档案 | 关 |

计划为 4 条合成语音×每条 3 次重复×4 组＝48 轮，共 12 个区组。同一区组使用相同音频，区组及组内执行顺序按种子 **20260913** 打乱。每轮新建会话，使用相同虚构档案与冻结记忆，关闭长期写入，逐轮核验状态哈希；单会话顺序负载，不测并发能力。

因提前终止，实际主分析只保留区组 **{', '.join(map(str, balanced['complete_block_ids']))}**，对应 `formal-001`～`formal-020`。完整区组筛选规则在额度中断后为公平比较而补充；以“四组均完整成功”为准，不按延迟大小挑样本，未改冻结的测量协议。只有 **{len(balanced['main_sample_block_counts'])} 条不同素材**进入主分析，并非计划的 4 条×3 次都完成。

| 合成素材 | 参考文本 | 裁剪后文件时长/s | 进入主分析的完整区组数 |
|---|---|---:|---:|
{md_rows(material_rows)}

正式结果不含 {all_summary['warmup_valid']} 轮有效预热；更早 4 轮开场状态重叠的预试另目录保留并排除。没有使用真实患者录音。

### 2.2 实际运行环境

- 预检时本机端口 8427、8502、8001、18019 的相关服务不可连接；当前 SoulX 配置无法用于实测。因此仅启动隔离测试实例 `127.0.0.1:18427`，使用独立测试 SQLite 库，不操作既有业务会话。
- VAD：真实 Silero VAD，沿用 `VAD_END_SILENCE_S=1.2`；未测 SoulX 语义轮次判断。
- ASR：真实火山 BigASR 流式识别，资源 `volc.seedasr.sauc.duration`；核验明确最终包，不把整段回退混入流式结果。
- 记忆：真实 SQLite 权威记忆卡开关；不是 Memobase 语义检索效果或检索时延消融。
- 情绪：真实 `emotion2vec_plus_large`，RTX 3090 GPU；以服务端最终元数据确认实际调用。
- LLM：DashScope `qwen3.7-flash`，temperature=0.65、max_tokens=240、thinking=OFF。输入配置相同，但开关改变提示上下文；回复内容和长度不强行锁定。
- TTS：真实火山 `seed-tts-2.0`，音色 `zh_female_vv_uranus_bigtts`。
- 使用现有 Python 3.11 环境；未重装依赖、未修改 `.env` 或业务源码、未重启其他服务。实验结束后仅停止隔离的 18427 测试实例，清理证据单独保存。

### 2.3 实时输入与计时零点

16 kHz 单声道 PCM16，每帧 512 个样本，即 **32 ms/帧**；按墙钟节奏发送二进制 WebSocket 音频，边发边接收事件，不是 `manual_audio` 整段提交。语音结束后发送静音，由真实 VAD 自行断句。

零点为**最后一个能量有效语音帧发送完成时刻**：帧 RMS 超过峰值帧 RMS 的 8%。这是可复算的帧/能量边界估计，不是物理人声结束或人工标注端点，存在 32 ms 帧量化及能量阈值误差。所有时间来自同一客户端单调时钟；loopback 不代表远程终端网络。

正式已发送输入帧相对计划的最大滞后 **{f(pace_max,3)} ms**；逐帧计划/实际发送时间全部保留。会话连接、开场合成和开场状态等待不计入“说完以后”的时间，因此本报告也不是从点击开始通话起算的首用时延。

## 3. 主消融结果：完整区组比较

主分析时间范围：{main_start}～{main_end}。以下全部基于 **{balanced['main_n']} 轮、每组 {n} 轮、相同 {n} 个区组**，单位为 ms。

| 配置 | n | 末帧→首段AI文本均值 | 末帧→首音频均值 | 首音频中位数 | 首音频最大值 | 首音频样本标准差 |
|---|---:|---:|---:|---:|---:|---:|
{md_rows(condition_rows)}

“首音频”指客户端收到首个非空回复音频块，回复的完整音频另经解码、有限值和非静音校验。**没有测到扬声器何时开始播放**，也不包含浏览器缓冲、声卡等。JSON 和交互图保留最近秩 P95，但本次每组 {n} 个样本的 P95 等于最大值，不作为稳定生产 P95。

### 3.1 同轮可相加分段

| 配置 | 语音末帧→VAD结束 | VAD结束→最终转写 | 最终转写→首段AI文本 | 首段AI文本→首非空音频 |
|---|---:|---:|---:|---:|
{md_rows(component_rows)}

上述四段来自同一轮、同一时钟，可相加核对总等待。两模块均开启组约为 **{f(full[COMPONENTS[0][0]]['mean']/1000)}＋{f(full[COMPONENTS[1][0]]['mean']/1000)}＋{f(full[COMPONENTS[2][0]]['mean']/1000)}＋{f(full[COMPONENTS[3][0]]['mean']/1000)}＝{f(e2e['mean']/1000)} s**（显示舍入允许毫秒级差异）。

- **VAD结束→最终转写不是 ASR 首字时延**。增量转写可能在用户说话时已出现；不可把它错配成 VAD_END 后的首字。
- 最终转写→首段AI文本包含实际 Agent 前处理、上下文/情绪处理、云端生成和文本分段，并非纯 LLM 推理时间或首 token 时间。
- 首段AI文本→首音频包含 TTS 路径与网络/事件传输，不是纯声学模型推理时间。
- 不同测试时钟、旧版手动提交、SoulX 转写或独立组件成绩均未混入本表。

### 3.2 TTS结束区间：只作描述，不再次相加

| 配置 | 首段AI文本→TTS结束均值/ms | 回复字数均值 | 生成音频时长均值/s |
|---|---:|---:|---:|
{md_rows(completion_rows)}

该区间包含后续文本生成与音频合成的重叠过程，受回复长度影响。**不能再与“首文本→首音频”相加**，TTS_END 也不是终端扬声器播完。

## 4. 消融差值与解释

同一区组内的首音频等待相减，正值表示左配置更慢，负值表示更快；全部仅用上述 {n} 个完整区组，单位 ms。

| 左配置 − 右配置 | 配对数 | 差值均值 | 差值中位数 | 最小差值 | 最大差值 |
|---|---:|---:|---:|---:|---:|
{md_rows(paired_rows)}

**当前不能得出“关闭记忆/情绪一定更快”的结论。** 单独关闭模块的均值并没有一致下降；双关闭相对均开启的差异也只有本次小样本观测意义。配置变化会改变提示和回复内容，云端网络、生成随机性、TTS分块以及调用时刻均可能影响时间；重复相同合成语音不等于独立用户场景。

因此不报告显著性结论，不把这些差异写成稳定的模块开销、普遍加速收益或心理支持效果。本次早停与调用时序有关，未完成素材尤其睡眠场景的缺失也限制外推。

当前可确认的优化方向是：**先检查约1.29 s的端点等待，以及首段文本和首音频路径，而不是仅凭此表删除记忆或情绪模块。** 如果试验缩短 VAD 静音阈值，必须同时统计误截断率、停顿容忍和重说率；本轮没有改此参数。

## 5. 真实性、开关与审计证据

- 主分析情绪开启组实际使用音频情绪模型 **{emotion_used}/{len(enabled)}**；关闭组明确 `source=disabled` **{emotion_disabled}/{len(disabled)}**。开启组元数据的情绪推理均值 **{f(emotion_ms['mean'])} ms**，不直接叠加到链路总时间。
- 按 session_id 关联的服务日志显示，记忆开启卡长度 **{', '.join(map(str, audited['actual_card_chars_by_arm']['M1E1']))} 字符**，关闭为 **{', '.join(map(str, audited['actual_card_chars_by_arm']['M0E0']))} 字符**（相同基础档案）。不把空语义检索写成命中。
- 记忆卡装载在会话准备阶段，主分析平均 **{f(card_ms['mean'])} ms**；不在语音末帧后这段等待内。这不是记忆动态检索速度测试。
- 档案/记忆哈希在 {accounting['started']} 个正式启动记录前后及最终审计时未变；已记录的 7 个业务源码哈希、`.env`、冻结测量脚本哈希均未变。
- 每个成功样本均满足单轮归属、最终转写、非空回复、实际有效音频及正常 TTS结束；配额失败与取消均不伪装成成功延迟。
- 审计 {audited['formal_records_audited']} 个正式记录，复算 {audited['recomputed_metric_count']} 项已存在的时间差；没有缺失时间补零，也没有把负数截断成0。
- 正式路径没有开场打断或流式失败整段回退。23项计时测试与15项新增汇总/早停测试，共38项全部通过；测试只验证逻辑，不冒充云端实验。

### 5.1 排除的开场预试

最初4次预试收到开场 TTS_END 后立即送语音，但服务在该事件之后才设置 `ai_speaking_until = now + duration`，导致触发打断和额外快速ASR/意图判断。这4轮另存于 `pilot_greeting_overlap_trials`，不混入正式统计。

正式协议等待开场合成结束后再等 duration＋0.25 s，使服务端状态自然结束；未发送伪造播放完成ACK，未将这段准备时间塞进语音末帧后延迟。这里只修正测量流程，未修改业务实现。

## 6. 全量成功描述与后续复测

下表是全部 **{len(good)} 个成功轮次**，样本量不平衡，**仅供追溯，不用于主消融比较**。额外成功记录为 `{extras}`；它们不是坏样本，只是所在区组四组未全部成功。原始全量汇总的逐对比较也可能纳入不完整区组，主结论只用独立的 `complete_block_summary.json`。

| 配置 | 全部成功数 | 末帧→首段AI文本均值/ms | 末帧→首音频均值/ms |
|---|---:|---:|---:|
{md_rows(all_rows)}

建议后续：
1. 先由账号负责人恢复现有 LLM 模型可用额度/权限。未获授权不变更计费设置；恢复后另开一轮完整随机配对实验，不把不同运行时段悄悄拼成一次完整48轮。
2. 在新的执行器版本补上 403配额错误熔断，首次明确配额拒绝即终止，避免重复失败调用。保留本轮冻结脚本与协议作为证据。
3. 如果报告对象是原型完整系统，应先恢复 SoulX 与 Memobase，再以相同语音、冻结档案、模块实际生效校验和计时定义复测；明确 SoulX 是否自带最终转写，避免混淆ASR来源。
4. 单独做 VAD 阈值、LLM首段分块、TTS首块的配对优化；补充更多人声/自然停顿素材及真实浏览器播放起点埋点。本轮不涉及终端听感、情绪识别准确率或临床效果。

## 7. 可复核文件

- [完整区组主分析]({out}/complete_block_summary.json)
- [所有启动记录的原始汇总]({out}/realtime_summary.json)
- [正式轮次成功/失败/取消账目]({out}/formal_accounting.json)
- [独立时间戳、开关与配额审计]({out}/measurement_audit.json)
- [冻结测量协议]({out}/realtime_protocol.json)
- [全部逐轮原始事件、逐帧发送记录、文本和音频]({out}/realtime_trials)
- [LLM配额错误证据]({out}/llm_quota_error_evidence.json)
- [提前停止记录]({out}/early_stop.json)
- [隔离部署预检与环境覆盖]({out}/preflight_plan.json)
- [测试服务清理与配置不变证明]({out}/cleanup.json)
- [计时脚本测试]({out}/measurement_pytest.log)
- [报告/早停逻辑测试]({out}/reporting_pytest.log)
- [38项联合回归测试]({out}/all_measurement_reporting_pytest.log)
- [测量脚本]({ROOT}/scripts/benchmark_voice_realtime_ablation.py)
- [审计脚本]({ROOT}/scripts/audit_voice_realtime_ablation.py)
- [离线报告生成脚本]({ROOT}/scripts/report_voice_realtime_ablation.py)
- [交互结果页]({canvas_path})

本报告没有修改之前的第六章 Word。全部样本、档案和记忆均为合成测试材料，账号凭证不写入交付物。
'''
    report_path.write_text(report, encoding='utf-8')
    (out / '实时语音消融实验复测报告.md').write_text(report, encoding='utf-8')
    data = {
        'start': start, 'end': end, 'mainStart': main_start, 'mainEnd': main_end,
        'terminated': audited['study_terminated_early'], 'accounting': accounting,
        'mainN': balanced['main_n'], 'perArm': n, 'blockIds': balanced['complete_block_ids'],
        'sampleCount': len(balanced['main_sample_block_counts']), 'auditCount': audited['recomputed_metric_count'],
        'cardOn': audited['actual_card_chars_by_arm']['M1E1'], 'cardOff': audited['actual_card_chars_by_arm']['M0E0'],
        'emotionOn': f'{emotion_used}/{len(enabled)}', 'emotionOff': f'{emotion_disabled}/{len(disabled)}',
        'fullMeanMs': e2e['mean'], 'maxPacingLagMs': pace_max, 'groups': [
            {'arm': arm, 'name': LABELS[arm], **g['metrics']['speech_end_to_first_audio_ms'],
             'firstText': g['metrics']['speech_end_to_first_ai_text_ms']['mean'],
             'components': [g['metrics'][k]['mean'] for k, _ in COMPONENTS]} for arm, g in groups.items()],
        'armAccounting': arm_accounting, 'pairedRows': paired_rows,
        'trials': [{'id': r['trial_id'], 'arm': r['arm'], 'block': r['block_id'],
            'state': '成功·主分析' if r['trial_id'] in main_ids else ('成功·仅附录' if r['valid_for_latency'] else DISPOSITIONS[by_audit[r['trial_id']]['disposition']]),
            'e2e': r['metrics'].get('speech_end_to_first_audio_ms'), 'asr': r.get('asr_text') or '未发送测量输入'} for r in records],
        'source': str(out / 'complete_block_summary.json'), 'report': str(report_path),
        'audit': str(out / 'measurement_audit.json'), 'quota': str(out / 'llm_quota_error_evidence.json'),
    }
    canvas_path.write_text(CANVAS_TEMPLATE.replace('__DATA__', json.dumps(data, ensure_ascii=False, allow_nan=False)), encoding='utf-8')
    manifest = {'status': 'terminated_early' if audited['study_terminated_early'] else 'schedule_executed',
        'report': str(report_path), 'canvas': str(canvas_path), 'main_analysis': str(out / 'complete_block_summary.json'),
        'all_records_summary': str(out / 'realtime_summary.json'), 'accounting': str(out / 'formal_accounting.json'),
        'audit': str(out / 'measurement_audit.json'), 'cleanup': str(out / 'cleanup.json')}
    save(out / 'deliverables.json', manifest)
    print(json.dumps({'deliverables': manifest, 'accounting': accounting, 'main_group_means_ms': {
        arm: {'first_text': g['metrics']['speech_end_to_first_ai_text_ms']['mean'],
              'first_audio': g['metrics']['speech_end_to_first_audio_ms']['mean']} for arm, g in groups.items()}}, ensure_ascii=False, indent=2))


CANVAS_TEMPLATE = '''import { useState, useHostTheme, H1, H2, Text, Stack, Row, Grid, Stat, Callout, Button, Table, BarChart, Link } from "cursor/canvas";

const data = __DATA__;
type Statistic = "mean" | "median" | "p95_nearest_rank";
const labels: Record<Statistic,string> = {mean:"均值",median:"中位数",p95_nearest_rank:"最近秩 P95（本样本等于最大值）"};
const fmt = (v: number | null | undefined) => typeof v === "number" ? v.toFixed(2) : "未产生";

export default function RealtimeVoiceAblation() {
  const theme = useHostTheme();
  const [stat, setStat] = useState<Statistic>("mean");
  return <main style={{padding:24,maxWidth:1120,margin:"0 auto",color:theme.text.primary}}><Stack gap={20}>
    <header><H1>实时语音消融 · 部分实测结果</H1>
      <Text tone="secondary">{data.start.slice(0,10)} · {data.start.slice(11,19)}—{data.end.slice(11,19)} 北京时间</Text></header>
    {data.terminated && <Callout tone="warning" title="提前停止：DashScope LLM 免费额度耗尽，不是火山 ASR/TTS 停用">
      计划 {data.accounting.planned} 轮；{data.accounting.successful} 轮完整成功，{data.accounting.llm_quota_failed} 轮 HTTP 403 配额拒绝，
      {data.accounting.cancelled_before_measured_input} 轮在准备阶段中止，{data.accounting.not_run} 轮未执行。未切换模型或开启付费，已停止请求并关闭隔离测试实例。
    </Callout>}
    <Row gap={28} wrap>
      <Stat value={`${fmt(data.fullMeanMs/1000)} s`} label="均开启组：末帧→收到首音频均值" tone="info" />
      <Stat value={`${data.mainN} 轮`} label={`主分析：每组 ${data.perArm} 轮，${data.perArm} 个完整区组`} />
      <Stat value={`${data.accounting.successful} 轮`} label="完整成功，额外2轮仅纳入附录" />
      <Stat value={data.auditCount} label="独立逐项复算一致的时间差" />
    </Row>
    <Text tone="secondary">本地 Silero VAD＋SQLite 记忆卡部署；真实流式 ASR、Emotion2Vec、LLM、TTS。不是 SoulX＋Memobase 完整系统，终点不是扬声器播放。</Text>
    <section><H2>语音末帧→客户端首个非空回复音频（ms）</H2>
      <Row gap={8} wrap>{(["mean","median","p95_nearest_rank"] as Statistic[]).map(k =>
        <Button key={k} variant={stat===k ? "primary" : "secondary"} onClick={()=>setStat(k)}>{labels[k]}</Button>)}</Row>
      <BarChart horizontal height={260} beginAtZero categories={data.groups.map(g=>g.name)}
        series={[{name:`末帧→首音频 ${labels[stat]}（ms）`,data:data.groups.map(g=>g[stat]),tone:"info"}]}
        valueSuffix=" ms" showValues />
      <Text size="small" tone="secondary">横轴：延迟（ms）；纵轴：消融配置。仅用区组 {data.blockIds.join("、")}，{data.sampleCount} 条不同语音，各组同为 {data.perArm} 轮。P95只是本次最大值，不代表生产尾延迟。</Text>
      <Text size="small" tone="tertiary">来源：{data.source} · {data.mainStart.slice(11,19)}—{data.mainEnd.slice(11,19)}；4轮有效预热及4轮开场重叠预试排除。</Text>
    </section>
    <section><H2>同轮分段均值 · 四段相加可核对首音频等待</H2>
      <Table headers={["配置","末帧→VAD / ms","VAD→最终转写 / ms","最终转写→首文本 / ms","首文本→首音频 / ms"]}
        rows={data.groups.map(g=>[g.name,...g.components.map(fmt)])} columnAlign={["left","right","right","right","right"]} />
      <Text size="small" tone="secondary">零点按32 ms帧、峰值RMS的8%估计，有帧/能量边界误差。最终转写不等于ASR首字；首文本不是纯LLM首token；TTS完成区间不能再次相加。</Text>
      <Text size="small" tone="tertiary">来源与时间：同上，20轮完整区组主分析，单位ms，算术均值。</Text>
    </section>
    <Grid columns={2} gap={24}>
      <section><H2>当前可以说什么</H2>
        <Text>均开启链路说完后约5.13秒收到首音频。等待来自端点静音判定、最终转写、首段回复和TTS首音频路径。</Text>
        <Text>关闭记忆或情绪没有一致变快；不能把这个小样本中的均值差异当作固定模块开销。先定位VAD、首文本和首音频路径。</Text>
      </section>
      <section><H2>不能外推的结论</H2>
        <Text>每组只有5轮，实际只覆盖3条合成语音，睡眠场景没有完整成功区组；不是4条语音各重复3次的完整结果。</Text>
        <Text tone="secondary">未测SoulX、Memobase语义检索、浏览器播放缓存、并发负载、临床效果或识别准确率。恢复同一LLM额度后应新开完整复测。</Text>
      </section>
    </Grid>
    <section><H2>原计划与实际去向（轮）</H2>
      <Table headers={["配置","计划","启动","成功","配额拒绝","中止","未执行","主分析"]}
        rows={data.armAccounting.map(g=>[g.label,g.planned,g.started,g.valid,g.quota,g.cancelled,g.not_run,g.main])} />
      <Text size="small" tone="tertiary">来源：原始协议、47个正式结果与按会话关联的服务日志；正式全时段。失败没有填0或从账目中删除。</Text>
    </section>
    <details><summary style={{cursor:"pointer",color:theme.text.secondary}}>查看配对首音频差值与实际开关证据</summary>
      <Table headers={["左配置 − 右配置","配对数","均值/ms","中位数/ms","最小/ms","最大/ms"]} rows={data.pairedRows} />
      <Text size="small" tone="secondary">正值表示左配置更慢；仅同一5个完整区组，不进行显著性宣称。</Text>
      <Table headers={["验证项","真实观测"]} rows={[
        ["主分析情绪开启：音频模型参与",data.emotionOn],["主分析情绪关闭：明确disabled",data.emotionOff],
        ["记忆开启 / 关闭实际卡长度",`${data.cardOn.join("/")} / ${data.cardOff.join("/")} 字符`],
        ["输入帧最大调度滞后",`${fmt(data.maxPacingLagMs)} ms`],
        ["冻结档案、记忆、已记录源码与.env","核验通过；业务配置未修改"]]} />
    </details>
    <details><summary style={{cursor:"pointer",color:theme.text.secondary}}>查看全部47个启动记录（含失败，不补造时间）</summary>
      <Table headers={["轮次","组别","区组","去向","末帧→首音频/ms","最终转写"]}
        rows={data.trials.map(r=>[r.id,r.arm,r.block,r.state,fmt(r.e2e),r.asr])} />
    </details>
    <Row gap={16} wrap><Link href={data.report}>完整中文报告</Link><Link href={data.source}>完整区组原始统计</Link>
      <Link href={data.audit}>独立审计</Link><Link href={data.quota}>配额错误证据</Link></Row>
  </Stack></main>;
}
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--canvas', type=Path, required=True)
    args = parser.parse_args()
    make(args.output, args.canvas)
