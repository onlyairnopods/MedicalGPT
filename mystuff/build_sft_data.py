import sys
import json
from datasets import load_dataset

reasoning_start = "<think>"
reasoning_end   = "</think>"
answer_start    = "<answer>"
answer_end      = "</answer>"

system_prompt = f"""Respond and think in the following format:
{reasoning_start}
...
{reasoning_end}
Then provide your final answer in the following format:
{answer_start}
...
{answer_end}
"""

dataset = load_dataset(
    "json",
    data_files="/cephfs/songyue/zzlai/hf_cache/hub/datasets--FreedomIntelligence--medical-o1-reasoning-SFT/snapshots/fc2c9e8a37b38f38da6d449564a8c350b244aef4/medical_o1_sft_mix_Chinese.json",
    split="train"
)

dataset = dataset.filter(
    lambda x: x["Question"] is not None and x["Complex_CoT"] is not None and x["Response"] is not None
)

def sft_format_dataset(x):
    problem = x["Question"].strip()
    thoughts = x["Complex_CoT"].replace("<think>", "").replace("</think>", "").strip()
    expected_answer = x["Response"].strip()

    final_answer = (
        f"{reasoning_start}\n"
        f"{thoughts}\n"
        f"{reasoning_end}\n"
        f"{answer_start}\n"
        f"{expected_answer}\n"
        f"{answer_end}"
    )

    return {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": problem},
            {"from": "gpt", "value": final_answer},
        ]
    }

def grpo_format_dataset(x):
    question = x["Question"].strip()
    answer = x["Response"].strip()
    return {
        "question": question,
        "answer": answer
    }


dataset = dataset.shuffle(seed=42)

split_idx = 10000
sft_dataset = dataset.select(range(split_idx))
rl_dataset = dataset.select(range(split_idx, len(dataset)))

sft_dataset = sft_dataset.map(sft_format_dataset, remove_columns=dataset.column_names)
rl_dataset = rl_dataset.map(grpo_format_dataset)

print(
    f"{sft_dataset[0]=}"
)
print(
    f"{rl_dataset[0]=}"
)
print(
    f"{len(sft_dataset)=}, {len(rl_dataset)=}"
)

sft_dataset.to_json("/cephfs/songyue/zzlai/MedicalGPT/myreasondata/mycoldsftdata/medical_o1_sft_mix_Chinese.jsonl", orient = "records", lines = True)

rl_dataset.to_json("/cephfs/songyue/zzlai/MedicalGPT/myreasondata/mygrpodata/medical_o1_rl_mix_Chinese.jsonl", orient = "records", lines = True)


conv_length_stat = []
for i in range(len(sft_dataset)):
    sys_prompt = sft_dataset[i]["conversations"][0]["value"]
    prompt = sft_dataset[i]["conversations"][1]["value"]
    response = sft_dataset[i]["conversations"][2]["value"]
    conv_length_stat.append(len(sys_prompt) + len(prompt) + len(response))

print(f"Average character length: {sum(conv_length_stat) / len(conv_length_stat)}")