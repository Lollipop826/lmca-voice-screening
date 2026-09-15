#!/usr/bin/env python3
"""Offline analysis of all successful formal realtime trials; no model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'M1E1': '记忆＋情绪开启', 'M0E1': '关闭记忆', 'M1E0': '关闭情绪', 'M0E0': '记忆、情绪均关闭'}
SAMPLES = {'memory-family-conflict-followup': '家庭联系', 'memory-community-walk-followup': '日常散步',
           'memory-unrelated-control': '忘带钥匙'}
AUDIO = 'speech_end_to_first_audio_ms'
TEXT = 'speech_end_to_first_ai_text_ms'
PARTS = [('speech_end_to_vad_end_ms', '语音末帧→VAD结束'),
         ('vad_end_to_asr_final_ms', 'VAD结束→最终转写'),
         ('asr_final_to_first_ai_text_ms', '最终转写→首段AI文本'),
         ('first_ai_text_to_first_audio_ms', '首段AI文本→首个回复音频块')]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    values = list(values)
    if not values or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        raise ValueError('Statistics require real, finite observations; no zero filling')
    return {'n': len(values), 'mean': round(statistics.fmean(values), 3),
            'median': round(statistics.median(values), 3), 'min': round(min(values), 3),
            'max': round(max(values), 3), 'std': round(statistics.stdev(values), 3) if len(values) > 1 else 0.0}


def aggregate(rows):
    metrics = set.intersection(*(set(r['metrics']) for r in rows))
    return {'n': len(rows), 'metrics': {k: stats(r['metrics'][k] for r in rows) for k in sorted(metrics)},
            'output_text_chars': stats(r['output_text_chars'] for r in rows),
            'output_audio_duration_s': stats(r['output_audio_duration_s'] for r in rows),
            'sample_counts': dict(Counter(r['sample_id'] for r in rows))}


def f(v, digits=1):
    return f'{v:.{digits}f}'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] + ['---:']*(len(headers)-1)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main(out: Path, canvas: Path):
    out, canvas = out.resolve(), canvas.resolve()
    audit = json.loads((out/'measurement_audit.json').read_text())
    audited = {r['trial_id']: r for r in audit['rows']}
    protocol = json.loads((out/'realtime_protocol.json').read_text())
    schedule = {r['trial_id']: r for r in protocol['schedule']}
    rows = []
    record_hashes = {}
    for path in sorted((out/'realtime_trials').glob('formal-*/result.json')):
        r = json.loads(path.read_text())
        if r['stage'] != 'formal' or r['valid_for_latency'] is not True:
            continue
        a = audited[r['trial_id']]
        if a['issues'] or not a['valid_for_latency'] or digest(path) != a['raw_sha256']['result.json']:
            raise ValueError('Selected success record differs from audited evidence')
        if any(r.get(k) != v for k, v in schedule[r['trial_id']].items()):
            raise ValueError('Selected record differs from frozen protocol')
        if not all(r['integrity_checks'].values()):
            raise ValueError('Incomplete record cannot enter successful-only analysis')
        for metric, value in r['metrics'].items():
            if abs(value - a['recomputed_metrics'][metric]) > .0006:
                raise ValueError('Original timestamp audit mismatch: ' + metric)
        rows.append(r)
        record_hashes[r['trial_id']] = a['raw_sha256']
    if len(rows) != 22 or len({r['trial_id'] for r in rows}) != 22:
        raise ValueError('This report is scoped to the requested 22 distinct successful formal trials')
    overall = aggregate(rows)
    groups = {arm: aggregate([r for r in rows if r['arm'] == arm]) for arm in LABELS}
    for g in groups.values():
        if abs(sum(g['metrics'][k]['mean'] for k, _ in PARTS) - g['metrics'][AUDIO]['mean']) > .004:
            raise ValueError('Mean decomposition is not additive')
    total = overall['metrics'][AUDIO]['mean']
    components = [{'key': k, 'label': label, **overall['metrics'][k],
                   'share_pct': round(100*overall['metrics'][k]['mean']/total, 3)} for k, label in PARTS]
    base = groups['M1E1']['metrics'][AUDIO]['mean']
    deltas = {arm: {'mean_difference_ms': round(g['metrics'][AUDIO]['mean'] - base, 3),
                    'relative_difference_pct': round(100*(g['metrics'][AUDIO]['mean'] - base)/base, 3)}
              for arm, g in groups.items()}
    slow = sorted((r for r in rows if r['arm'] == 'M1E0'), key=lambda r:r['metrics'][AUDIO], reverse=True)[:2]
    start, end = min(r['started_at'] for r in rows), max(r['finished_at'] for r in rows)
    selected = [{'trial_id': r['trial_id'], 'arm': r['arm'], 'block_id': r['block_id'], 'sample_id': r['sample_id'],
                 'started_at': r['started_at'], 'finished_at': r['finished_at'], 'metrics': r['metrics'],
                 'output_text_chars': r['output_text_chars'], 'output_audio_duration_s': r['output_audio_duration_s']}
                for r in rows]
    analysis = {'schema_version': 'successful-realtime-trials-v1', 'created_at': datetime.now().astimezone().isoformat(),
        'selection': 'All successful formal trials (valid_for_latency=true), including formal-021 and formal-022; no block-completeness exclusion, trimming or replacement',
        'n': len(rows), 'start': start, 'end': end, 'overall': overall, 'groups': groups,
        'component_means_and_shares': components, 'unadjusted_differences_vs_M1E1': deltas,
        'interpretation': 'Descriptive latency conditional on successful completion; unbalanced group/sample counts; not causal module costs or an availability analysis',
        'scope': 'Silero local VAD / SQLite memory card / Volcengine streaming ASR and TTS / Emotion2Vec / DashScope LLM; client received audio, not speaker playback',
        'audited_metric_count': sum(len(audited[r['trial_id']]['recomputed_metrics']) for r in rows),
        'selected_record_hashes': record_hashes, 'trials': selected}
    summary_path = out/'successful_trials_summary.json'
    summary_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    report_path = ROOT/'docs/realtime_voice_successful22_analysis_20260913.md'
    group_rows = [[LABELS[arm], g['n'], f(g['metrics'][TEXT]['mean']), f(g['metrics'][AUDIO]['mean']),
                   f(g['metrics'][AUDIO]['median']), f(g['metrics'][AUDIO]['std']),
                   f(g['metrics'][AUDIO]['min'])+'～'+f(g['metrics'][AUDIO]['max'])] for arm,g in groups.items()]
    component_rows = [[c['label'], f(c['mean']), f(c['share_pct'])+'%'] for c in components]
    group_parts = [[LABELS[arm]]+[f(g['metrics'][k]['mean']) for k,_ in PARTS] for arm,g in groups.items()]
    counts = '、'.join(f'{SAMPLES[s]}{n}轮' for s,n in overall['sample_counts'].items())
    slow_text = '、'.join(f"`{r['trial_id']}`（{f(r['metrics'][AUDIO])} ms）" for r in slow)
    quote = (f"在本地Silero VAD与SQLite记忆卡部署下，对22轮成功完成的实时语音交互进行统计。"
        f"用户有效语音末帧发送后，客户端收到首段AI文本的平均等待为{f(overall['metrics'][TEXT]['mean'],2)} ms，"
        f"收到首个回复音频块的平均等待为{f(total,2)} ms，中位数为{f(overall['metrics'][AUDIO]['median'],2)} ms。"
        f"记忆与情绪均开启、关闭记忆、关闭情绪和双关闭配置的首音频平均等待分别为"
        f"{f(base,2)}、{f(groups['M0E1']['metrics'][AUDIO]['mean'],2)}、"
        f"{f(groups['M1E0']['metrics'][AUDIO]['mean'],2)}和{f(groups['M0E0']['metrics'][AUDIO]['mean'],2)} ms。"
        "本次未观察到单独禁用记忆或情绪即可稳定降低响应延迟的趋势。由于各组样本量为5或6轮，"
        "且素材分布略有差异，这些结果用于描述本次成功交互的性能，不能解释为模块的独立耗时或统计显著的性能收益。")
    report = f'''# 实时语音消融实验分析：22轮成功样本

## 1. 分析口径与总体表现

本版**完整纳入22轮成功的正式实验**，即 `formal-001`～`formal-022`，包括之前没有进入完整区组主表的 `formal-021` 和 `formal-022`。不再将主分析限制为20轮；不做失败原因或可用率分析，不删去较慢的成功样本。

- 成功样本时间：{start}～{end}（北京时间）。
- 四组样本量依次为 **5、6、5、6轮**；共有3条不同合成语音，{counts}。
- **语音末帧→首段AI文本：平均 {f(overall['metrics'][TEXT]['mean'])} ms（{f(overall['metrics'][TEXT]['mean']/1000,2)}秒）。**
- **语音末帧→首个回复音频：平均 {f(total)} ms（{f(total/1000,2)}秒），中位数 {f(overall['metrics'][AUDIO]['median'])} ms。**
- 首音频等待的实际范围为 **{f(overall['metrics'][AUDIO]['min'])}～{f(overall['metrics'][AUDIO]['max'])} ms**，样本标准差 {f(overall['metrics'][AUDIO]['std'])} ms。

总体均值直接按22个轮次计算，不是对四个组均值简单平均。以上仅描述“成功交互条件下的延迟”。

## 2. 四组消融结果

下表所有时间单位为 **ms**，均使用该组全部成功样本。

{table(['配置','成功轮数','末帧→首文本均值','末帧→首音频均值','首音频中位数','首音频标准差','首音频范围'],group_rows)}

### 2.1 对消融结果的解释

1. **均开启配置平均约{f(base/1000,2)}秒；双关闭配置平均约{f(groups['M0E0']['metrics'][AUDIO]['mean']/1000,2)}秒。** 双关闭组的原始均值低约 **{f(-deltas['M0E0']['mean_difference_ms'])} ms（{f(-deltas['M0E0']['relative_difference_pct'])}%）**。这是观测差异，不能称为“记忆与情绪模块只耗时这些毫秒”。
2. **单独关闭记忆或情绪，没有显示一致的提速趋势。** 相对均开启组，关闭记忆的均值高 {f(deltas['M0E1']['mean_difference_ms'])} ms，关闭情绪高 {f(deltas['M1E0']['mean_difference_ms'])} ms。这不能反向证明开启模块会加速；提示上下文、首段回复、网络和合成时序都可能改变。
3. **关闭情绪组波动最大，不宜只看均值。** 该组首音频均值 {f(groups['M1E0']['metrics'][AUDIO]['mean'])} ms，但中位数为 {f(groups['M1E0']['metrics'][AUDIO]['median'])} ms；较慢的两轮为 {slow_text}，均被保留。均开启组的中位数反而高于关闭情绪组，说明均值排序并不等于稳定排序。
4. **样本量与素材分布必须保留说明。** 记忆开启的两组各有家庭2轮、钥匙2轮、散步1轮；记忆关闭的两组各有三类素材各2轮。表中比较为不作素材校正的描述性统计，不能将约百毫秒的差异直接当作因果效应，也不进行显著性宣称。

## 3. 约5.15秒主要花在哪里

以下按全部22轮求平均，四段来自每轮同一客户端单调时钟，可相加得到“末帧→首音频”。

{table(['阶段','22轮均值/ms','占首音频总等待'],component_rows)}

**首段AI文本生成相关阶段与TTS首音频路径合计约 {f(components[2]['share_pct']+components[3]['share_pct'])}%**；端点等待约 {f(components[0]['share_pct'])}%。最终转写衔接约 {f(components[1]['mean']/1000,2)}秒，并不是这条链路中最长的一段。

各组的四段均值如下，单位ms：

{table(['配置','末帧→VAD结束','VAD→最终转写','最终转写→首文本','首文本→首音频'],group_parts)}

### 3.1 必须区分的时间定义

- “说完”以最后一个能量有效语音帧发送完成时刻估计：32 ms帧，阈值为峰值帧RMS的8%。存在帧量化和能量阈值误差，不是物理人声端点测量。
- **VAD结束→最终转写不是ASR首字时延。** 增量文字可能在用户还在说话时已返回。
- **最终转写→首段AI文本不等于纯LLM推理耗时**，包括实际上下文/Agent处理、情绪相关路径、生成与文本分段。
- **首音频指客户端收到音频，不是扬声器开始播放。** 不包含浏览器解码、播放缓存和声卡的启动延迟。
- 若查看TTS完成区间，它包含文本生成和合成的重叠过程，不能再与首音频等待重复相加。

## 4. 基于这22轮的建议

- **不要仅为这约百毫秒的观测差异删除记忆或情绪模块。** 本次数据没有证明这种做法会稳定提速，也没有测模块带来的回复质量变化。
- **优先检查端点与首包路径。** 当前VAD静音阈值为1.2秒，端点段实测均值约{f(components[0]['mean']/1000,2)}秒；首段文本和首个音频块还各需约1.6～1.7秒。可分别评估VAD阈值、回复分句策略和TTS首块策略，不能把它们混为单一模型延迟。
- **优化必须兼顾交互正确性。** 缩短VAD等待要同时统计误截断率；缩短首文本/首音频等待要检查语义完整性、断句和音频连续性。本分析未执行新的参数试验。

### 可用于报告的结论段落

> {quote}

## 5. 测量范围与22轮明细

实际部署为**本地Silero VAD＋SQLite权威记忆卡＋火山流式ASR＋Emotion2Vec＋DashScope LLM＋火山TTS**。它不是SoulX＋Memobase完整系统成绩，也不是远程浏览器或物理播放端到端时延。输入为合成语音，按32 ms/帧实时发送；未使用真实患者录音。

22条记录的 **{analysis['audited_metric_count']} 个时间差**均与独立原始时间戳审计一致。这里只离线重算已有成功样本，没有新增云端调用；原始记录没有改动。

{table(['轮次','配置','合成素材','末帧→首文本/ms','末帧→首音频/ms'],[[r['trial_id'],LABELS[r['arm']],SAMPLES[r['sample_id']],f(r['metrics'][TEXT]),f(r['metrics'][AUDIO])] for r in rows])}

- [仅含22轮成功样本的统计与逐轮数据]({summary_path})
- [22轮交互分析页]({canvas})
- [离线分析生成脚本]({ROOT}/scripts/report_voice_successful_trials.py)
'''
    report_path.write_text(report, encoding='utf-8')
    (out/'22轮成功样本实时延迟分析.md').write_text(report, encoding='utf-8')
    data = {'n':len(rows), 'start':start, 'end':end, 'overall':overall, 'components':components,
            'groups':[{'arm':arm,'label':LABELS[arm],**g} for arm,g in groups.items()],
            'trials':[{'id':r['trial_id'],'arm':r['arm'],'sample':SAMPLES[r['sample_id']],
                       'text':r['metrics'][TEXT],'audio':r['metrics'][AUDIO]} for r in rows],
            'auditCount':analysis['audited_metric_count'],'report':str(report_path),'source':str(summary_path)}
    canvas.write_text(CANVAS.replace('__DATA__', json.dumps(data, ensure_ascii=False, allow_nan=False)), encoding='utf-8')
    manifest={'report':str(report_path),'canvas':str(canvas),'summary':str(summary_path),'n':len(rows),'source_records_unchanged':True}
    (out/'successful_trials_deliverables.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'deliverables':manifest,'overall':overall['metrics'][AUDIO],'component_means_and_shares':components},ensure_ascii=False,indent=2))


CANVAS = '''import { useState, useHostTheme, H1, H2, Text, Row, Stack, Grid, Stat, Callout, Table, BarChart, Button, Link } from "cursor/canvas";
const data = __DATA__;
type Statistic = "mean" | "median";
const metric = "speech_end_to_first_audio_ms";
const textMetric = "speech_end_to_first_ai_text_ms";
const fmt = (n:number) => n.toFixed(1);
export default function SuccessfulVoiceTrials() {
  const theme = useHostTheme();
  const [stat,setStat] = useState<Statistic>("mean");
  return <main style={{padding:24,maxWidth:1120,margin:"0 auto",color:theme.text.primary}}><Stack gap={20}>
    <header><H1>22轮成功语音交互 · 延迟与消融分析</H1>
      <Text tone="secondary">{data.start.slice(0,10)} · {data.start.slice(11,19)}—{data.end.slice(11,19)} 北京时间</Text></header>
    <Callout tone="info" title="全部22轮成功样本均纳入，不再只取20轮">
      四组分别5、6、5、6轮，覆盖3条不同合成语音。仅描述成功交互的延迟；不删除较慢样本，不作可用率分析。
    </Callout>
    <Row gap={30} wrap>
      <Stat value={`${(data.overall.metrics[metric].mean/1000).toFixed(2)} s`} label="22轮：语音末帧→首音频均值" tone="info" />
      <Stat value={`${(data.overall.metrics[textMetric].mean/1000).toFixed(2)} s`} label="22轮：语音末帧→首段AI文本均值" />
      <Stat value={`${(data.overall.metrics[metric].median/1000).toFixed(2)} s`} label="22轮：首音频等待中位数" />
    </Row>
    <Text tone="secondary">测量终点为客户端收到首个回复音频块，不是扬声器播放。实际为Silero VAD＋SQLite记忆卡部署，不是SoulX＋Memobase完整系统。</Text>
    <section><H2>四组首音频等待：语音末帧→收到回复音频（ms）</H2>
      <Row gap={8}>{(["mean","median"] as Statistic[]).map(s=><Button key={s} variant={s===stat?"primary":"secondary"} onClick={()=>setStat(s)}>{s==="mean"?"均值":"中位数"}</Button>)}</Row>
      <BarChart horizontal height={260} beginAtZero categories={data.groups.map(g=>`${g.label}（n=${g.n}）`)}
        series={[{name:`首音频等待${stat==="mean"?"均值":"中位数"}（ms）`,data:data.groups.map(g=>g.metrics[metric][stat]),tone:"info"}]} valueSuffix=" ms" showValues />
      <Text size="small" tone="secondary">横轴：延迟（ms）；纵轴：消融配置与成功样本数。素材分布略有不同，均值差不等于模块独立开销。</Text>
      <Text size="small" tone="tertiary">来源：{data.source} · {data.start.slice(11,19)}—{data.end.slice(11,19)}，全部22轮。</Text>
    </section>
    <section><H2>四组完整数值（ms）</H2>
      <Table headers={["配置","n","首文本均值","首音频均值","首音频中位数","首音频标准差"]}
        rows={data.groups.map(g=>[g.label,g.n,fmt(g.metrics[textMetric].mean),fmt(g.metrics[metric].mean),fmt(g.metrics[metric].median),fmt(g.metrics[metric].std)])} />
      <Text size="small" tone="tertiary">来源与时间同上；标准差为样本标准差。所有值均按各组真实轮次统计。</Text>
    </section>
    <Grid columns={2} gap={24}>
      <section><H2>总体时间分配</H2>
        <Table headers={["阶段","均值/ms","占总等待"]} rows={data.components.map(c=>[c.label,fmt(c.mean),`${fmt(c.share_pct)}%`])} />
        <Text size="small" tone="secondary">四段同轮同钟可相加。首段文本与首音频路径合计约63.7%；最终转写不是ASR首字。</Text>
        <Text size="small" tone="tertiary">来源：同上，全部22轮分段算术均值；占比＝分段均值÷总等待均值。</Text>
      </section>
      <section><H2>应该怎样解读</H2>
        <Text>均开启约5.13秒，双关闭约5.03秒，原始均值相差约108 ms（2.1%）。不能据此认定两个模块的独立开销就是108 ms。</Text>
        <Text>单独关闭记忆或情绪没有一致变快。关闭情绪组均值5.27秒、中位数5.07秒，两个较慢样本拉高了均值，未做剔除。</Text>
        <Text tone="secondary">优先核查约1.29秒的端点等待，以及各约1.6～1.7秒的首文本和首音频路径；缩短VAD阈值必须兼顾误截断率。</Text>
      </section>
    </Grid>
    <details><summary style={{cursor:"pointer",color:theme.text.secondary}}>查看全部22轮成功数据</summary>
      <Table headers={["轮次","配置","素材","末帧→首文本/ms","末帧→首音频/ms"]}
        rows={data.trials.map(r=>[r.id,r.arm,r.sample,fmt(r.text),fmt(r.audio)])} />
    </details>
    <Text size="small" tone="secondary">{data.auditCount}个时间差与独立原始时间戳审计一致；本次仅离线重算，未新增云端调用。语音零点为32 ms帧、8%峰值RMS规则的估计，不是物理发声端点。</Text>
    <Row gap={16} wrap><Link href={data.report}>22轮成功样本分析报告</Link><Link href={data.source}>仅含22轮的统计与明细</Link></Row>
  </Stack></main>;
}
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--canvas', required=True, type=Path)
    args = parser.parse_args()
    main(args.output, args.canvas)
