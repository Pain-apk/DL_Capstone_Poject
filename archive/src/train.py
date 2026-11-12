"""
train.py - QLoRA fine-tuning for recipe generation using Qwen2.5-1.5B-Instruct.

Supports:
--smoke-test: Runs 50 steps, checks continuation gates (VRAM < 7.2 GB, loss decreasing, no errors).
--full-run: Runs full 2 epochs with checkpointing every 200 steps and resume capability.
"""

import os
import sys
import json
import time
import argparse
import datetime
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "pilot_unverified")
SMOKE_LOG = os.path.join(OUTPUT_DIR, "smoke_test_metrics.json")
LOSS_LOG = os.path.join(OUTPUT_DIR, "training_loss.log")

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

class VRAMAndLossLoggingCallback(TrainerCallback):
    def __init__(self, log_filepath: str):
        self.log_filepath = log_filepath
        self.step_losses = []
        self.peak_vram_mb = 0.0

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            if vram_mb > self.peak_vram_mb:
                self.peak_vram_mb = vram_mb

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            step = state.global_step
            loss = logs["loss"]
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
            self.step_losses.append((step, loss))
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{timestamp}] Step: {step:<5} | Loss: {loss:.4f} | Peak VRAM: {vram_mb:.1f} MiB\n"
            print(line.strip())
            with open(self.log_filepath, "a", encoding="utf-8") as f:
                f.write(line)


def format_chat_examples(batch, tokenizer, max_length=1024):
    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for messages in batch["messages"]:
        # Apply Qwen chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        enc = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Mask user and system tokens in labels
        # Find where assistant message begins
        labels = list(input_ids)
        # In Qwen template, assistant turn starts with <|im_start|>assistant\n
        assistant_tag = "<|im_start|>assistant\n"
        encoded_tag = tokenizer.encode(assistant_tag, add_special_tokens=False)
        tag_len = len(encoded_tag)

        # Locate tag in input_ids
        split_idx = -1
        for i in range(len(input_ids) - tag_len):
            if input_ids[i:i+tag_len] == encoded_tag:
                split_idx = i + tag_len
                break

        if split_idx != -1:
            for i in range(split_idx):
                labels[i] = -100
        # Mask padding
        for i in range(len(attention_mask)):
            if attention_mask[i] == 0:
                labels[i] = -100

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list
    }


def train_model(smoke_test: bool = False, resume_from_checkpoint: str = None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_data_path = os.path.join(PROJECT_ROOT, "data", "processed", "pilot_unverified", "train.jsonl")
    val_data_path = os.path.join(PROJECT_ROOT, "data", "processed", "pilot_unverified", "val.jsonl")

    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Training dataset not found: {train_data_path}. Run build_dataset.py first.")

    print(f"\nInitializing QLoRA fine-tuning for {MODEL_ID}")
    print(f"Mode: {'SMOKE TEST (50 steps)' if smoke_test else 'FULL 2-EPOCH TRAINING'}")

    # 1. 4-bit Quantization Config (NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # 2. Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Model
    print("Loading base model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # 4. LoRA Config (r=16, alpha=32 on all linear projections)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Dataset loading and tokenization
    print("Loading and tokenizing dataset...")
    raw_dataset = load_dataset(
        "json",
        data_files={"train": train_data_path, "val": val_data_path}
    )

    tokenized_dataset = raw_dataset.map(
        lambda batch: format_chat_examples(batch, tokenizer, max_length=1024),
        batched=True,
        remove_columns=raw_dataset["train"].column_names
    )

    # 6. Training Arguments
    log_file = LOSS_LOG if not smoke_test else os.path.join(OUTPUT_DIR, "smoke_loss.log")
    logging_callback = VRAMAndLossLoggingCallback(log_file)

    max_steps = 50 if smoke_test else -1
    num_train_epochs = 1 if smoke_test else 2

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10 if smoke_test else 50,
        logging_steps=5,
        save_strategy="steps",
        save_steps=200 if not smoke_test else 50,
        save_total_limit=2,
        eval_strategy="no" if smoke_test else "steps",
        eval_steps=200,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
        dataloader_num_workers=0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"] if not smoke_test else None,
        callbacks=[logging_callback]
    )

    # Pre-training VRAM and estimates
    if torch.cuda.is_available():
        init_vram = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"Base model loaded. VRAM in use: {init_vram:.1f} MiB")

    start_time = time.time()
    print("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    elapsed_sec = time.time() - start_time

    peak_vram = logging_callback.peak_vram_mb
    print(f"\nTraining completed in {elapsed_sec:.1f}s. Peak VRAM: {peak_vram:.1f} MiB")

    # Save metrics
    if smoke_test:
        losses = [l for _, l in logging_callback.step_losses]
        first_losses = losses[:3] if len(losses) >= 3 else losses
        last_losses = losses[-3:] if len(losses) >= 3 else losses
        initial_avg = sum(first_losses) / len(first_losses) if first_losses else 0.0
        final_avg = sum(last_losses) / len(last_losses) if last_losses else 0.0
        loss_decreasing = final_avg < initial_avg
        vram_pass = peak_vram < 7372.0  # 7.2 GB in MiB

        smoke_metrics = {
            "peak_vram_mb": round(peak_vram, 2),
            "peak_vram_gb": round(peak_vram / 1024.0, 3),
            "vram_gate_passed": bool(vram_pass),
            "initial_loss_avg": round(initial_avg, 4),
            "final_loss_avg": round(final_avg, 4),
            "loss_decreasing_gate_passed": bool(loss_decreasing),
            "training_errors": None,
            "overall_gate_passed": bool(vram_pass and loss_decreasing),
            "elapsed_seconds": round(elapsed_sec, 2),
            "sec_per_step": round(elapsed_sec / 50.0, 3)
        }
        with open(SMOKE_LOG, "w", encoding="utf-8") as f:
            json.dump(smoke_metrics, f, indent=2)

        print("\n" + "=" * 60)
        print("SMOKE TEST GATE EVALUATION:")
        print("=" * 60)
        print(f"1. Peak VRAM: {peak_vram:.1f} MiB / 7,372 MiB limit -> {'PASS' if vram_pass else 'FAIL'}")
        print(f"2. Loss trend: {initial_avg:.4f} -> {final_avg:.4f} -> {'PASS' if loss_decreasing else 'FAIL'}")
        print(f"3. Errors: None -> PASS")
        print(f"Overall Continuation Gate: {'PASSED - PROCEEDING' if smoke_metrics['overall_gate_passed'] else 'FAILED - STOPPING'}")
        print("=" * 60 + "\n")
        return smoke_metrics["overall_gate_passed"]
    else:
        # Save final LoRA adapter
        adapter_path = os.path.join(OUTPUT_DIR, "adapter")
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        print(f"Full pilot fine-tuning complete! LoRA adapter saved to: {adapter_path}")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run 50-step smoke test")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train_model(smoke_test=args.smoke_test, resume_from_checkpoint=args.resume)

