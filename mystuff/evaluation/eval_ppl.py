import warnings
warnings.filterwarnings("ignore")
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import argparse
parser = argparse.ArgumentParser(description="Compute assistant-only perplexity for chat models")
parser.add_argument("--model_name_or_path", "--model", type=str, default="Qwen/Qwen3-4B",
                    help="Model name or local path")
parser.add_argument("--dataset_name", type=str, default="FreedomIntelligence/Huatuo26M-Lite",
                    help="Dataset name")
parser.add_argument("--num_samples", type=int, default=10000,
                    help="Number of samples to evaluate")
parser.add_argument("--output_dir", type=str, default="./results",
                    help="Directory to save results")

args = parser.parse_args()

model_id = args.model_name_or_path
dataset_name = args.dataset_name
num_samples = args.num_samples
output_dir = args.output_dir

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", local_files_only=True)

tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

from datasets import load_dataset

dataset = load_dataset(dataset_name)
dataset = dataset['train'].map(
    lambda sample: {
        "conversations": [
                {"from": "human", "value": sample['question']},
                {"from": "gpt", "value": sample['answer']}
            ]
        },
    batched=False
)
dataset = dataset.select(range(num_samples))

import transformers
from typing import Dict, Sequence, List
from torch.utils.data import Dataset
from dataclasses import dataclass
from jinja2 import Template

def get_template(tokenizer):
    chat_template = tokenizer.chat_template
    if chat_template is None:
        return None
    return Template(chat_template)

template = get_template(tokenizer)
def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    template: Template = None,
) -> Dict:
    max_seq_len = tokenizer.model_max_length
    messages = []
    for i, source in enumerate(sources):
        if source[0]["from"] != "human":
            # Skip the first one if it is not from human
            source = source[1:]

        for j in range(0, len(source), 2):
            if j+1 >= len(source): continue
            q = source[j]["value"]
            a = source[j+1]["value"]
            assert q is not None and a is not None, f'q:{q} a:{a}'

            if template is not None:
                input = template.render(
                    messages=[{"role": "user", "content": q},{"role": "assistant", "content": a}],
                    bos_token=tokenizer.bos_token,
                    add_generation_prompt=False
                )
                input_ids = tokenizer.encode(input, add_special_tokens=False, padding_side="right")

                query = template.render(
                    messages=[{"role": "user", "content": q}],
                    bos_token=tokenizer.bos_token,
                    add_generation_prompt=True
                )
                query_ids = tokenizer.encode(query, add_special_tokens=False)
            else:
                # Base model: no chat template, just concat tokens
                q_ids = tokenizer.encode(q, add_special_tokens=False)
                a_ids = tokenizer.encode(a, add_special_tokens=False)
                input_ids = q_ids + a_ids
                query_ids = q_ids

            labels = [-100] * len(query_ids) + input_ids[len(query_ids):]
            assert len(labels) == len(input_ids)
            if len(input_ids) == 0: continue

            messages.append({"input_ids": input_ids[-max_seq_len:], "labels": labels[-max_seq_len:]})

    input_ids = [item["input_ids"] for item in messages]
    labels = [item["labels"] for item in messages]

    max_len = max(len(x) for x in input_ids)
    max_len = min(max_len, max_seq_len)

    input_ids = [ item[:max_len] + [tokenizer.eos_token_id] * (max_len - len(item)) for item in input_ids]
    labels = [ item[:max_len] + [-100] * (max_len-len(item)) for item in labels]

    input_ids = torch.LongTensor(input_ids)
    labels = torch.LongTensor(labels)
    return {
        "input_ids": input_ids,
        "labels": labels
    }


class InstructDataset(Dataset):
    def __init__(self, data: Sequence, tokenizer: transformers.PreTrainedTokenizer, template: Template = None) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data
        self.template = template

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        sources = self.data[index]
        if isinstance(index, int):
            sources = [sources]
        data_dict = preprocess([e['conversations'] for e in sources], self.tokenizer, self.template)
        if isinstance(index, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        return data_dict


IGNORE_INDEX = -100
@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

test_dataset = InstructDataset(dataset, tokenizer, template)
data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

trainer = transformers.Trainer(
    model=model,
    eval_dataset=test_dataset,
    data_collator=data_collator
)
import math
eval_results = trainer.evaluate()
ppl = math.exp(eval_results['eval_loss'])
print(f"Perplexity: {ppl:.2f}")


model_safe = model_id.rstrip("/").split("/")[-1]
result_dir = os.path.join(output_dir, model_safe)
os.makedirs(result_dir, exist_ok=True)

result_file = os.path.join(result_dir, "perplexity.json")
result = {
    "model_id": model_id,
    "dataset": dataset_name,
    "num_samples": num_samples,
    "perplexity": ppl,
}
import json
if os.path.exists(result_file):
    with open(result_file, "r", encoding="utf-8") as f:
        existing = json.load(f)
else:
    existing = {}

existing[dataset_name] = result

with open(result_file, "w", encoding="utf-8") as f:
    json.dump(existing, f, indent=2, ensure_ascii=False)

print(f"结果已保存到: {result_file}")