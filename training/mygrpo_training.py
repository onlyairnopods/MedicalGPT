import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Callable
import re
from datasets import load_dataset
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser
from peft import LoraConfig, TaskType, get_peft_model

from transformers import AutoModelForSequenceClassification

try:
    import flash_attn  # noqa: F401

    is_flash_attn_2_available = True
except ImportError:
    is_flash_attn_2_available = False

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with GRPO
    """
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The tokenizer for weights initialization."}
    )
    # Dataset arguments
    dataset_name: Optional[str] = field(
        default="openai/gsm8k",
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory containing training files for local datasets."}
    )
    train_samples: Optional[int] = field(default=-1, metadata={"help": "Number of samples to train on, -1 for all"})
    subset_name: Optional[str] = field(default="main",
                                       metadata={"help": "Subset name, e.g., 'default', 'main'. default is 'default'"})
    dataset_splits: Optional[str] = field(default="train", metadata={"help": "Split name"})
    preprocessing_num_workers: Optional[int] = field(default=10,
                                                     metadata={"help": "Number of workers for preprocessing"})
    
    # QLoRA arguments
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})


def extract_completion_text(completion) -> str:
    try:
        return completion[0]["content"].strip()
    except Exception:
        return ""


def extract_answer_content(text: str) -> str:
    ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    m = ANSWER_RE.search(text)
    if not m:
        return ""
    return m.group(1).strip()


def match_format_exactly_reward(completions, **kwargs):
    """1.0 if full strict format is correct, else 0.0"""
    STRICT_FORMAT_RE = re.compile(
        r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
        re.DOTALL,
    )
    rewards = []
    for completion in completions:
        content = extract_completion_text(completion)
        rewards.append(1.0 if STRICT_FORMAT_RE.match(content) else 0.0)
    logger.debug(f"exact format rewards: {rewards}")
    return rewards


def match_format_approximately_reward(completions, **kwargs):
    """
    Soft format score in [0, 1].
    Reward:
    - correct tag counts
    - correct order
    - answer content exists
    """
    scores = []
    for completion in completions:
        text = extract_completion_text(completion)

        score = 0.0

        think_open = text.count("<think>")
        think_close = text.count("</think>")
        answer_open = text.count("<answer>")
        answer_close = text.count("</answer>")

        score += 0.2 if think_open == 1 else 0.0
        score += 0.2 if think_close == 1 else 0.0
        score += 0.2 if answer_open == 1 else 0.0
        score += 0.2 if answer_close == 1 else 0.0

        think_pos = text.find("<think>")
        think_end_pos = text.find("</think>")
        answer_pos = text.find("<answer>")
        answer_end_pos = text.find("</answer>")

        order_ok = (
            think_pos != -1
            and think_end_pos != -1
            and answer_pos != -1
            and answer_end_pos != -1
            and think_pos < think_end_pos < answer_pos < answer_end_pos
        )
        score += 0.1 if order_ok else 0.0

        answer_content = extract_answer_content(text)
        score += 0.1 if answer_content else 0.0

        scores.append(score)

    logger.debug(f"approx format rewards: {scores}")
    return scores

class SemanticCorrectnessCalculator:
    """
    Cross-encoder for response-vs-reference semantic scoring.
    """

    def __init__(self, model_name: str = 'BAAI/bge-reranker-v2-m3', device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=True,
        ).to(device).eval()


@torch.inference_mode()
def semantic_correctness_reward(
    calculator: SemanticCorrectnessCalculator,
    responses: list[str],
    answers: list[str],
    max_length: int = 512,
) -> list[float]:
    if len(responses) != len(answers):
        raise ValueError("responses and answers must have the same length")

    if not responses:
        return []

    pairs = list(zip(responses, answers))
    inputs = calculator.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=max_length)
    scores = calculator.model(**inputs, return_dict=True).logits.view(-1, ).float()

    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    else:
        scores = list(scores)

    # 空响应直接打最低
    scores = [-1.0 if resp == "" else float(score) for resp, score in zip(responses, scores)]
    return scores


class PerplexityCalculator:
    """
    Optional LM fluency scorer.
    """

    def __init__(self, model_name: str, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,
            **kwargs
        ).eval()
        self.device = self.model.device


@torch.inference_mode()
def perplexity_reward(
    calculator: PerplexityCalculator,
    texts: list[str],
    batch_size: int = 8,
    max_length: int = 512,
) -> list[float]:
    """
    返回每条样本自己的 perplexity。
    越低越好。
    """
    all_ppls = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if not batch:
            continue

        safe_batch = [t if t.strip() else calculator.tokenizer.eos_token for t in batch]

        try:
            enc = calculator.tokenizer(
                safe_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(calculator.device)
            attention_mask = enc["attention_mask"].to(calculator.device)

            outputs = calculator.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            label_mask = attention_mask[:, 1:]

            token_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="none",
            ).view(labels.size())

            token_loss = token_loss * label_mask
            valid_token_counts = label_mask.sum(dim=1).clamp(min=1)
            seq_nll = token_loss.sum(dim=1) / valid_token_counts
            seq_ppl = torch.exp(seq_nll).detach().cpu().tolist()

            for raw_text, ppl in zip(batch, seq_ppl):
                all_ppls.append(1e6 if not raw_text.strip() else float(ppl))

        except Exception as e:
            logger.warning(f"Perplexity batch failed at batch {i // batch_size}: {e}")
            all_ppls.extend([1e6] * len(batch))

    return all_ppls


def normalize_similarity_scores(similarities: list[float]) -> torch.Tensor:
    """
    尽量把不同模型输出 squash 到 [0, 1]
    """
    sims = torch.tensor(similarities, dtype=torch.float32)
    sims = torch.nan_to_num(sims, nan=-1.0)

    sim_min = sims.min().item()
    sim_max = sims.max().item()

    # 常见情况 1：本来就在 [0, 1]
    if sim_min >= 0.0 and sim_max <= 1.0:
        return sims

    # 常见情况 2：在 [-1, 1]
    if sim_min >= -1.0 and sim_max <= 1.0:
        return (sims + 1.0) / 2.0

    # 其他情况：用 sigmoid 压缩
    return torch.sigmoid(sims)


def normalize_ppl_to_reward(ppls: list[float]) -> torch.Tensor:
    """
    ppl 越低越好，映射到 [0, 1]
    """
    ppl = torch.tensor(ppls, dtype=torch.float32)
    ppl = torch.nan_to_num(ppl, nan=1e6, posinf=1e6, neginf=1e6)

    # log1p 稳一点
    ppl = torch.log1p(ppl)

    score_range = ppl.max() - ppl.min()
    if score_range < 1e-6:
        return torch.ones_like(ppl) * 0.5

    # ppl 小 -> reward 大
    reward = 1.0 - (ppl - ppl.min()) / score_range
    return reward


def make_combined_reward_func(
    semantic_calculator: Optional[PerplexityCalculator] = None,
    perplexity_calculator: Optional[PerplexityCalculator] = None,
    semantic_weight: float = 0.8,
    format_weight: float = 0.19,
    ppl_weight: float = 0.01,
) -> Callable:
    """
    推荐默认：
    - semantic_weight=0.8
    - format_weight=0.20
    - ppl_weight=0.0
    """
    total = semantic_weight + format_weight + ppl_weight

    # 归一化权重
    semantic_weight /= total
    format_weight /= total
    ppl_weight /= total

    def combined_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        n = len(completions)
        final_rewards = [-1.0] * n

        # 先给全部 completion 算格式分
        exact_format_scores = torch.tensor(
            match_format_exactly_reward(completions),
            dtype=torch.float32,
        )
        approx_format_scores = torch.tensor(
            match_format_approximately_reward(completions),
            dtype=torch.float32,
        )
        format_scores = 0.5 * exact_format_scores + 0.5 * approx_format_scores  # [0, 1]

        valid_positions = []
        responses = []
        ref_answers = []

        for idx, completion in enumerate(completions):
            generated_text = extract_completion_text(completion)
            generated_answer = extract_answer_content(generated_text)

            if not generated_answer or not generated_answer.strip():
                continue

            user_prompt = ""
            try:
                user_prompt = prompts[idx][-1]["content"].strip()
            except Exception:
                pass

            if user_prompt and generated_answer.strip() == user_prompt:
                continue

            valid_positions.append(idx)
            responses.append(generated_answer.strip())
            ref_answers.append(answer[idx].strip())

        # 没有有效答案，直接返回
        if not valid_positions:
            logger.debug("No valid completions found.")
            return final_rewards

        try:
            if semantic_calculator is not None and semantic_weight > 0:
                semantic_raw = semantic_correctness_reward(
                    semantic_calculator,
                    responses,
                    ref_answers,
                )
                semantic_scores = normalize_similarity_scores(semantic_raw)
                logger.debug(f"semantic scores: {semantic_scores.tolist()}")
            else:
                semantic_scores = torch.zeros(len(valid_positions), dtype=torch.float32)

            if perplexity_calculator is not None and ppl_weight > 0:
                # 注意：这里是对生成答案算 ppl，不是参考答案
                ppls = perplexity_reward(perplexity_calculator, responses)
                ppl_scores = normalize_ppl_to_reward(ppls)
                logger.debug(f"ppl scores: {ppl_scores.tolist()}")
            else:
                ppl_scores = torch.zeros(len(valid_positions), dtype=torch.float32)

        except Exception as e:
            logger.warning(f"Reward calculation error: {e}")
            return final_rewards

        for local_i, global_i in enumerate(valid_positions):
            reward = (
                semantic_weight * semantic_scores[local_i].item()
                + format_weight * format_scores[global_i].item()
                + ppl_weight * ppl_scores[local_i].item()
            )

            # map [0,1] -> [-1,1]
            reward = reward * 2.0 - 1.0
            reward = max(min(float(reward), 1.0), -1.0)
            final_rewards[global_i] = reward

            logger.debug(f"predict_answer: {responses[local_i]}, \nground_truth: {ref_answers[local_i]}\n"
                     f"reward: {reward}\n\n")

        logger.debug(f"final combined rewards: {final_rewards}")
        return final_rewards

    return combined_reward_func




SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def grpo_train(
        model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    # Add distributed training initialization
    is_main_process = training_args.local_rank in [-1, 0]

    # Only log on main process
    if is_main_process:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        logger.info(f"Model parameters {model_args}")
        logger.info(f"Script parameters {script_args}")
        logger.info(f"Training parameters {training_args}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    if script_args.train_file_dir and os.path.exists(script_args.train_file_dir):
        # Load from local directory
        dataset = load_dataset("json", data_dir=script_args.train_file_dir, split="train")
    else:
        # Load from HuggingFace hub
        dataset = load_dataset(script_args.dataset_name, script_args.subset_name, split=script_args.dataset_splits)

    if script_args.train_samples > 0:
        dataset = dataset.shuffle(seed=42).select(range(script_args.train_samples))

    # Prepare dataset
    with training_args.main_process_first(desc="Dataset preparation"):
        dataset = dataset.map(
            lambda x: {
                'prompt': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': x['question']}
                ],
                'answer': x['answer']
            },
            num_proc=script_args.preprocessing_num_workers,
            desc="Processing dataset" if is_main_process else None,
        )

    # Split dataset
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    if is_main_process:
        logger.info("*** Initializing model kwargs ***")

    # Model initialization
    torch_dtype = (
        model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    )

    # Set up distributed training config
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1

    # Check for QLoRA compatibility
    if script_args.qlora and is_deepspeed_zero3_enabled():
        logger.warning("ZeRO3 are both currently incompatible with QLoRA.")

    # Check quantization settings
    if model_args.load_in_4bit and model_args.load_in_8bit:
        raise ValueError("Error, load_in_4bit and load_in_8bit cannot be set at the same time")

    # Set up quantization config
    quantization_config = None
    if script_args.qlora and (model_args.load_in_4bit or model_args.load_in_8bit):
        if is_main_process:
            logger.info(
                f"Quantizing model, load_in_4bit: {model_args.load_in_4bit}, load_in_8bit: {model_args.load_in_8bit}")
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif model_args.load_in_4bit or model_args.load_in_8bit:
        # Support quantization even without qlora flag
        if is_main_process:
            logger.info(
                f"Quantizing model, load_in_4bit: {model_args.load_in_4bit}, load_in_8bit: {model_args.load_in_8bit}")
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=torch_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        quantization_config=quantization_config,
    )

    num_gpus = torch.cuda.device_count()
    if ddp:
        model_kwargs["device_map"] = None
    elif num_gpus > 1:
        max_memory = {}
        for i in range(num_gpus):
            gpu_props = torch.cuda.get_device_properties(i)
            total_mem = gpu_props.total_memory
            # 预留20%内存给训练时的梯度、优化器状态等
            usable_mem = int(total_mem * 0.8)
            max_memory[i] = f"{usable_mem // (1024 ** 3)}GiB"
        model_kwargs["max_memory"] = max_memory
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"

    if is_main_process:
        logger.info(f"Using {num_gpus} GPUs")
        logger.info(f"model_kwargs={model_kwargs}")

    config = AutoModelForCausalLM.config_class if hasattr(AutoModelForCausalLM, 'config_class') else None
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            revision=model_args.model_revision
        )
    except Exception:
        config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    # Patch MoE modules for DeepSpeed ZeRO-3
    model_type = getattr(config, "model_type", None) if config else getattr(model.config, "model_type", None)
    if model_type == "mixtral" and is_deepspeed_zero3_enabled():
        from deepspeed.utils import set_z3_leaf_modules
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
        set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

    if model_type == "deepseek_v3" and is_deepspeed_zero3_enabled():
        for layer in model.model.layers:
            if 'DeepseekV3MoE' in str(type(layer.mlp)):
                layer.mlp._z3_leaf = True

    if model_type == "qwen3_moe" and is_deepspeed_zero3_enabled():
        from deepspeed.utils import set_z3_leaf_modules
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
        set_z3_leaf_modules(model, [Qwen3MoeSparseMoeBlock])

    if model_type == "qwen3_5_moe" and is_deepspeed_zero3_enabled():
        from deepspeed.utils import set_z3_leaf_modules
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
        set_z3_leaf_modules(model, [Qwen3_5MoeSparseMoeBlock])

    if is_main_process and hasattr(model, 'hf_device_map'):
        logger.info(f"Model Device Map: {model.hf_device_map.items()}")
    elif is_main_process and num_gpus > 1:
        logger.info("Model Device Map:")
        for name, param in model.named_parameters():
            if hasattr(param, 'device'):
                logger.info(f"  {name}: {param.device}")
                break

    # Configure LoRA if enabled
    if model_args.use_peft:
        if is_main_process:
            logger.info("Fine-tuning method: LoRA(PEFT)")
        if training_args.gradient_checkpointing:
            logger.warning("Gradient checkpointing is enabled. It may cause issues with LoRA, setting it to False.")
            training_args.gradient_checkpointing = False
        target_modules = model_args.lora_target_modules if model_args.lora_target_modules else None
        if target_modules == 'all' or (target_modules and 'all' in target_modules):
            target_modules = find_all_linear_names(model, int4=model_args.load_in_4bit, int8=model_args.load_in_8bit)
        if is_main_process:
            logger.info(f"Peft target_modules: {target_modules}, lora rank: {model_args.lora_r}, ")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
        )
        model = get_peft_model(model, peft_config)
        # Fixed FP16 ValueError for quantized models
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
        model.print_trainable_parameters()
    else:
        if is_main_process:
            logger.info("Fine-tuning method: Full parameters training")

    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        logger.info("Gradient checkpointing enabled.")
    else:
        model.config.use_cache = True
        logger.info("Gradient checkpointing disabled.")

    semantic_calculator = SemanticCorrectnessCalculator(
        model_name="/cephfs/songyue/zzlai/hf_cache/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        device=model_kwargs["device_map"],
    )
    # perplexity_calculator = PerplexityCalculator(
    #     model_name="/cephfs/songyue/zzlai/hf_cache/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987",
    #     device_map=model_kwargs["device_map"],
    #     attn_implementation=model_args.attn_implementation,
    #     dtype=torch_dtype,
    #     low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
    # )
    perplexity_calculator = None

    reward_func = make_combined_reward_func(
        semantic_calculator=semantic_calculator,
        perplexity_calculator=perplexity_calculator,
        semantic_weight=0.8,
        format_weight=0.20,
        ppl_weight=0.0,
    )

    # Initialize GRPO trainer with distributed training support
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_func,
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if training_args.eval_strategy != "no" else None,
    )
    logger.info("*** GRPO Trainer initialized ***")
    logger.debug(f"Trainer: {trainer}")

    # Training
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        if is_main_process:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    if is_main_process:
        logger.info(
            f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for '
            f'{training_args.num_train_epochs} epochs ***'
        )

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Log and save metrics on main process
    if is_main_process:
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info("*** Training complete ***")
        logger.info("*** Save model ***")

    # Save model
    trainer.model.config.use_cache = True
    if is_main_process:
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

    training_args.distributed_state.wait_for_everyone()

    if is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"Tokenizer saved to {training_args.output_dir}")

        # Create model card and save config
        kwargs = {
            "dataset_name": script_args.dataset_name,
            "tags": ["r1", "grpo"],
        }
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if is_main_process:
        logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_train(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
