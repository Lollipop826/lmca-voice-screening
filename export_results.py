import json
import os
import pandas as pd

files = [
    "e2e_full.json",
    "e2e_no_memory.json",
    "e2e_no_emotion.json",
    "e2e_latency.json",
]

rows = []

for f in files:
    path = "output/" + f

    with open(path, "r") as fp:
        data = json.load(fp)

    summary = data.get("summary", {})

    row = {
        "experiment": f,
        "asr_ms_mean": summary.get("input_to_asr_result_ms", {}).get("mean_ms"),
        "ai_first_ms_mean": summary.get("input_to_first_ai_text_ms", {}).get("mean_ms"),
        "tts_first_byte_ms_mean": summary.get("input_to_first_tts_byte_ms", {}).get("mean_ms"),
        "tts_end_ms_mean": summary.get("input_to_tts_end_ms", {}).get("mean_ms"),
    }

    rows.append(row)

df = pd.DataFrame(rows)

df.to_excel(
    "output/experiment_results.xlsx",
    index=False
)

print(df)
