#!/usr/bin/env python3
"""Isolate whether the companion agent can use a correct emotion label.

The test reuses frozen, memory-disabled prompts from the 112-response experiment.
For each prosody scenario it compares the original neutral label with the
scenario's prespecified oracle label.  It does not test Emotion2Vec itself.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import random
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.wellbeing_companion_agent import (
    WellbeingCompanionAgent,
    _emotion_control_block,
)


SOURCE = ROOT / "output/real_voice_ablation_20260914_quality_sensitive"
OUT = ROOT / "output/oracle_emotion_label_pilot_v4_20260914"
PROMPTS = SOURCE / "agent_prompt_events.jsonl"
PROTOCOL = SOURCE / "realtime_protocol.json"
LABEL_PREFIX = "【内部情绪推测·仅供参考·不得在回复中提及·与对方原话冲突时以原话为准】"
DIMENSIONS = ("emotion_fit", "empathy", "calibration", "helpfulness", "naturalness")
REPEATS = 2
JUDGE_REPEATS = 3


def now() -> str:
    return datetime.now().astimezone().isoformat()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response is not a JSON object")
    return value


def api_call(
    *, endpoint: str, api_key: str, body: dict[str, Any], timeout: float = 50.0
) -> tuple[str, float]:
    last_error: Exception | None = None
    for attempt in range(1, 5):
        started = time.perf_counter()
        try:
            session = requests.Session()
            session.trust_env = False
            response = session.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=body,
                timeout=timeout,
            )
            if response.status_code == 400 and "response_format" in body:
                body = dict(body)
                body.pop("response_format", None)
                response = session.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=body,
                    timeout=timeout,
                )
            response.raise_for_status()
            payload = response.json()
            content = str(payload["choices"][0]["message"]["content"])
            return content, round((time.perf_counter() - started) * 1000, 1)
        except Exception as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(attempt)
    raise RuntimeError(f"API failed: {type(last_error).__name__}: {last_error}")


def load_cases() -> list[dict[str, str]]:
    wanted = {
        "prosody-achievement-happy": "joy",
        "prosody-achievement-sad": "sadness",
        "prosody-boundary-anger": "anger",
        "prosody-boundary-neutral": "calm",
        "prosody-plan-fear": "fear",
        "prosody-plan-neutral": "calm",
    }
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    by_id = {row["sample_id"]: row for row in protocol["samples"]}
    result = []
    for sample_id, label in wanted.items():
        row = by_id[sample_id]
        result.append(
            {
                "sample_id": sample_id,
                "text": row["reference_text"],
                "target_label": label,
                "expected_behavior": row["expected_behavior"],
                "provenance": "scenario-prespecified oracle label; not a human-validated audio annotation",
            }
        )
    return result


def load_prompt_templates(cases: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    templates: dict[str, list[dict[str, str]]] = {}
    agent = WellbeingCompanionAgent(llm_factory=lambda: None)
    for case in cases:
        messages = agent._build_messages(
            case["text"],
            {"name": "虚构评测用户", "age": 68, "gender": "female", "education_years": 9},
            [],
            "neutral",
        )
        templates[case["sample_id"]] = [
            {
                "role": "user" if message.type == "human" else message.type,
                "content": str(message.content),
            }
            for message in messages
        ]
    return templates


def replace_label(messages: list[dict[str, str]], label: str) -> list[dict[str, str]]:
    copied = json.loads(json.dumps(messages, ensure_ascii=False))
    pattern = re.compile(
        r"【内部语音情绪控制信号·严禁向用户披露】.*?"
        r"不要推断原因、对象、强度或用户未表达的经历。",
        flags=re.S,
    )
    for message in copied:
        if message["role"] == "system":
            message["content"] = WellbeingCompanionAgent._SYSTEM_PROMPT
        if message["role"] == "user" and "【内部语音情绪控制信号" in message["content"]:
            changed, count = pattern.subn(_emotion_control_block(label), message["content"], count=1)
            if count != 1:
                raise ValueError("emotion label replacement failed")
            message["content"] = changed
            return copied
    raise ValueError("emotion label line not found")


def generate_one(task: dict[str, Any], endpoint: str, api_key: str, model: str) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": task["messages"],
        "temperature": 0.45,
        "max_tokens": 240,
    }
    content, elapsed_ms = api_call(endpoint=endpoint, api_key=api_key, body=body)
    return {
        "generation_id": task["generation_id"],
        "sample_id": task["sample_id"],
        "repeat": task["repeat"],
        "condition": task["condition"],
        "injected_label": task["injected_label"],
        "text": task["text"],
        "target_label": task["target_label"],
        "expected_behavior": task["expected_behavior"],
        "response": content.strip(),
        "model": model,
        "elapsed_ms": elapsed_ms,
        "completed_at": now(),
    }


def build_judge_prompt(task: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是心理陪伴对话的匿名评审。A/B来自同一模型、同一用户文字和同一上下文；"
                "唯一可能差异是内部情绪信号，但你不知道哪一个使用了什么信号。"
                "按目标语音情绪和预期行为评价，不因回复直接说出情绪名称而加分；"
                "若回复武断臆测用户情绪，应降低校准度。只输出JSON。"
            ),
        },
        {
            "role": "user",
            "content": f"""用户文字：{task['text']}
目标语音情绪（由实验场景预设）：{task['target_label']}
预期行为：{task['expected_behavior']}

回答A：{task['responses']['A']}

回答B：{task['responses']['B']}

请分别按1到5整数评分：
- emotion_fit：与目标语音情绪的匹配度
- empathy：共情与接住感受的质量
- calibration：不过度解读、不武断假设
- helpfulness：对当前需要的帮助程度
- naturalness：自然、简洁、像熟人聊天

最后给总体偏好A、B或tie。严格输出：
{{"A":{{"emotion_fit":1,"empathy":1,"calibration":1,"helpfulness":1,"naturalness":1}},"B":{{"emotion_fit":1,"empathy":1,"calibration":1,"helpfulness":1,"naturalness":1}},"preference":"A","reason":"一句具体依据"}}""",
        },
    ]


def judge_one(task: dict[str, Any], endpoint: str, api_key: str, model: str) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": build_judge_prompt(task),
        "temperature": 0,
        "max_tokens": 900,
        "response_format": {"type": "json_object"},
    }
    content, elapsed_ms = api_call(endpoint=endpoint, api_key=api_key, body=body)
    value = parse_json_content(content)
    for label in ("A", "B"):
        scores = value.get(label)
        if not isinstance(scores, dict) or set(scores) != set(DIMENSIONS):
            raise ValueError(f"invalid score dimensions for {label}")
        if any(type(scores[d]) is not int or not 1 <= scores[d] <= 5 for d in DIMENSIONS):
            raise ValueError(f"invalid scores for {label}")
    if value.get("preference") not in {"A", "B", "tie"}:
        raise ValueError("invalid preference")
    return {
        "judge_id": task["judge_id"],
        "pair_id": task["pair_id"],
        "sample_id": task["sample_id"],
        "generation_repeat": task["generation_repeat"],
        "judge_repeat": task["judge_repeat"],
        "target_label": task["target_label"],
        "mapping": task["mapping"],
        "scores": {"A": value["A"], "B": value["B"]},
        "preference": value["preference"],
        "reason": str(value.get("reason") or "")[:1000],
        "model": model,
        "elapsed_ms": elapsed_ms,
        "completed_at": now(),
    }


def append_parallel(
    tasks: list[dict[str, Any]], function: Any, progress_path: Path, id_key: str,
    endpoint: str, api_key: str, model: str, workers: int,
) -> list[dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row[id_key]] = row
    pending = [task for task in tasks if task[id_key] not in done]
    print(json.dumps({"stage": id_key, "total": len(tasks), "done": len(done), "pending": len(pending)}, ensure_ascii=False), flush=True)
    with progress_path.open("a", encoding="utf-8", buffering=1) as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(function, task, endpoint, api_key, model): task for task in pending}
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                done[row[id_key]] = row
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[{id_key}] {len(done)}/{len(tasks)} {row[id_key]}", flush=True)
    return [done[task[id_key]] for task in tasks]


def bootstrap_ci(values: list[float], seed: int, rounds: int = 10000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(rounds))
    return [round(means[int(rounds * 0.025)], 3), round(means[int(rounds * 0.975)], 3)]


def analyze(cases: list[dict[str, str]], generations: list[dict[str, Any]], ratings: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in ratings:
        by_pair.setdefault(row["pair_id"], []).append(row)
    pair_rows = []
    for pair_id, rows in sorted(by_pair.items()):
        deltas = {d: [] for d in DIMENSIONS}
        prefs = {"oracle": 0, "neutral": 0, "tie": 0}
        for row in rows:
            inv = {condition: label for label, condition in row["mapping"].items()}
            for dim in DIMENSIONS:
                deltas[dim].append(row["scores"][inv["oracle"]][dim] - row["scores"][inv["neutral"]][dim])
            pref = row["preference"]
            prefs["tie" if pref == "tie" else row["mapping"][pref]] += 1
        pair_rows.append(
            {
                "pair_id": pair_id,
                "sample_id": rows[0]["sample_id"],
                "target_label": rows[0]["target_label"],
                "mean_delta_oracle_minus_neutral": {d: round(statistics.mean(v), 3) for d, v in deltas.items()},
                "preference_votes": prefs,
            }
        )

    def subset_summary(selected: list[dict[str, Any]], seed_offset: int) -> dict[str, Any]:
        result: dict[str, Any] = {"n_generation_pairs": len(selected), "dimensions": {}}
        for i, dim in enumerate(DIMENSIONS):
            values = [row["mean_delta_oracle_minus_neutral"][dim] for row in selected]
            result["dimensions"][dim] = {
                "mean_delta": round(statistics.mean(values), 3),
                "median_delta": round(statistics.median(values), 3),
                "bootstrap_95ci": bootstrap_ci(values, 91400 + seed_offset + i),
            }
        votes = {"oracle": 0, "neutral": 0, "tie": 0}
        for row in selected:
            for key, value in row["preference_votes"].items():
                votes[key] += value
        result["preference_votes_across_repeated_calls"] = votes
        return result

    nonneutral = [row for row in pair_rows if row["target_label"] not in {"calm", "neutral"}]
    calm = [row for row in pair_rows if row["target_label"] in {"calm", "neutral"}]
    return {
        "created_at": now(),
        "design": {
            "cases": len(cases),
            "generation_repeats": REPEATS,
            "generation_pairs": len(pair_rows),
            "judge_repeats_per_pair": JUDGE_REPEATS,
            "note": "judge repeats are repeated calls to one model, not independent human raters",
            "causal_scope": "tests downstream agent use of a prespecified correct label; does not test emotion recognition accuracy",
        },
        "all_cases": subset_summary(pair_rows, 0),
        "nonneutral_cases": subset_summary(nonneutral, 100),
        "calm_cases": subset_summary(calm, 200),
        "pair_results": pair_rows,
        "generation_count": len(generations),
        "rating_count": len(ratings),
    }


def write_report(analysis: dict[str, Any]) -> None:
    names = {
        "emotion_fit": "情绪匹配",
        "empathy": "共情",
        "calibration": "校准/不过度解读",
        "helpfulness": "帮助性",
        "naturalness": "自然度",
    }
    lines = [
        "# 正确情绪标签（Oracle）注入小实验",
        "",
        "> 本实验把场景预设的正确标签直接送入 Agent，并与通用 `neutral` 标签比较。标签不是人工听音标注，因此结果只用于故障定位，不替代正式语音情绪识别评测。",
        "",
        "## 汇总（Oracle − neutral）",
        "",
        "| 范围 | 生成配对数 | 指标 | 平均分差 | 95% Bootstrap CI |",
        "|---|---:|---|---:|---:|",
    ]
    for section, label in (("all_cases", "全部6场景"), ("nonneutral_cases", "非中性4场景"), ("calm_cases", "平静2场景")):
        row = analysis[section]
        for dim in DIMENSIONS:
            metric = row["dimensions"][dim]
            ci = metric["bootstrap_95ci"]
            lines.append(f"| {label} | {row['n_generation_pairs']} | {names[dim]} | {metric['mean_delta']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    lines += ["", "## 偏好票", ""]
    for section, label in (("all_cases", "全部"), ("nonneutral_cases", "非中性"), ("calm_cases", "平静")):
        votes = analysis[section]["preference_votes_across_repeated_calls"]
        lines.append(f"- {label}：Oracle {votes['oracle']}，neutral {votes['neutral']}，平局 {votes['tie']}。")
    emotion = analysis["nonneutral_cases"]["dimensions"]["emotion_fit"]
    prefs = analysis["nonneutral_cases"]["preference_votes_across_repeated_calls"]
    if emotion["mean_delta"] > 0.5 and prefs["oracle"] > prefs["neutral"]:
        conclusion = (
            "本轮观察到方向明确的提升；但样本量仍小，情绪匹配置信区间跨过0，"
            "应把它视为通过小样本验收，而不是正式显著性结论。"
        )
    else:
        conclusion = "本轮没有观察到正确标签带来的稳定提升。"
    lines += [
        "",
        "## 初步结论",
        "",
        (
            f"非中性场景的情绪匹配平均分差为 {emotion['mean_delta']:+.3f} 分，"
            f"95% Bootstrap CI 为 [{emotion['bootstrap_95ci'][0]:+.3f}, {emotion['bootstrap_95ci'][1]:+.3f}]；"
            f"总体偏好票 Oracle/neutral 为 {prefs['oracle']}/{prefs['neutral']}。"
            + conclusion
        ),
    ]
    lines += [
        "",
        "## 解释边界",
        "",
        "- 若非中性场景明显提升，说明 Agent 能利用正确标签，主要瓶颈在语音情绪刺激/识别端。",
        "- 若非中性场景仍无提升，说明 Agent 提示词对情绪标签的利用也偏弱。",
        "- 评分来自同一个 Qwen 模型的重复匿名调用，不等同于3名独立评审；正式报告仍建议补人工盲评。",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = dotenv_values(ROOT / ".env")
    api_key = str(env.get("SILICONFLOW_API_KEY") or "")
    base_url = str(env.get("SILICONFLOW_BASE_URL") or "https://api.siliconflow.cn/v1")
    model = str(env.get("SILICONFLOW_MODEL") or "qwen-flash")
    if not api_key:
        raise ValueError("SILICONFLOW_API_KEY is not configured")
    endpoint = base_url.rstrip("/") + "/chat/completions"

    cases = load_cases()
    templates = load_prompt_templates(cases)
    generation_tasks = []
    for case in cases:
        for repeat in range(1, REPEATS + 1):
            for condition, label in (("neutral", "neutral"), ("oracle", case["target_label"])):
                generation_tasks.append(
                    {
                        **case,
                        "generation_id": f"{case['sample_id']}__r{repeat}__{condition}",
                        "repeat": repeat,
                        "condition": condition,
                        "injected_label": label,
                        "messages": replace_label(templates[case["sample_id"]], label),
                    }
                )
    dump_json(
        OUT / "protocol.json",
        {
            "created_at": now(),
            "source": str(SOURCE),
            "model": model,
            "cases": cases,
            "generation_repeats": REPEATS,
            "judge_repeats": JUDGE_REPEATS,
            "generation_temperature": 0.45,
            "memory": "disabled in frozen prompt template",
            "conditions": {"neutral": "neutral label", "oracle": "scenario-prespecified correct label"},
        },
    )
    generations = append_parallel(
        generation_tasks, generate_one, OUT / "generations.jsonl", "generation_id",
        endpoint, api_key, model, 6,
    )
    dump_json(OUT / "generations.json", generations)
    generation_by_id = {row["generation_id"]: row for row in generations}

    judge_tasks = []
    by_case = {row["sample_id"]: row for row in cases}
    for case in cases:
        for repeat in range(1, REPEATS + 1):
            pair_id = f"{case['sample_id']}__r{repeat}"
            pair = {
                condition: generation_by_id[f"{case['sample_id']}__r{repeat}__{condition}"]["response"]
                for condition in ("neutral", "oracle")
            }
            for judge_repeat in range(1, JUDGE_REPEATS + 1):
                seed = int(hashlib.sha256(f"{pair_id}:{judge_repeat}:914".encode()).hexdigest()[:12], 16)
                labels = ["A", "B"]
                random.Random(seed).shuffle(labels)
                mapping = {labels[0]: "neutral", labels[1]: "oracle"}
                judge_tasks.append(
                    {
                        **by_case[case["sample_id"]],
                        "judge_id": f"{pair_id}__j{judge_repeat}",
                        "pair_id": pair_id,
                        "generation_repeat": repeat,
                        "judge_repeat": judge_repeat,
                        "mapping": mapping,
                        "responses": {label: pair[condition] for label, condition in mapping.items()},
                    }
                )
    dump_json(
        OUT / "blind_judge_sheet.json",
        [{k: v for k, v in row.items() if k != "mapping"} for row in judge_tasks],
    )
    dump_json(OUT / "hidden_key.json", [{"judge_id": row["judge_id"], "mapping": row["mapping"]} for row in judge_tasks])
    ratings = append_parallel(
        judge_tasks, judge_one, OUT / "ratings.jsonl", "judge_id",
        endpoint, api_key, model, 6,
    )
    dump_json(OUT / "ratings.json", ratings)
    result = analyze(cases, generations, ratings)
    dump_json(OUT / "analysis.json", result)
    write_report(result)
    print(json.dumps({"status": "complete", "output": str(OUT), "summary": result["nonneutral_cases"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
