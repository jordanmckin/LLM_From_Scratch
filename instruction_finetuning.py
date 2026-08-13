import tiktoken
import torch
import os
import sys
import random
import time
import json
from tqdm import tqdm
from datasets import load_dataset
from InstructionFinetuningDataLoader import InstructionDataset, custom_collate_fn, format_input, create_instruction_dataloader



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tokenizers import Tokenizer
from Transformer import TransformerModel
from YUNA_GPT import generate
from YUNA_Training import text_to_token_ids, token_ids_to_text, train_model_simple, get_linear_lr_scheduler
from YUNA_CONFIG import YUNA_GPT_CONFIG
from YUNA_CONFIG import OTHER_SETTINGS




def save_instruction_data(data, out_path="instruction_data.jsonl"):
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(data)} examples to {out_path}")

def load_instruction_data(path="instruction_data.jsonl"):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def estimate_instruction_tokens(data_path="instruction_data.jsonl", sample_fraction=0.1):
    tokenizer = Tokenizer.from_file("yuna_bpe.json")
    data = load_instruction_data(data_path)

    n_sample = max(1, int(len(data) * sample_fraction))
    sample = data[:n_sample]  # or random.sample(data, n_sample) for a more representative pull

    total_tokens = 0
    for entry in tqdm(sample, desc="Sampling instruction tokens"):
        instruction_plus_input = format_input(entry)
        response_text = f"\n\n### Response:\n{entry['output']}"
        full_text = instruction_plus_input + response_text
        total_tokens += len(tokenizer.encode(full_text).ids)

    avg_tokens_per_example = total_tokens / n_sample
    estimated_total_tokens = int(avg_tokens_per_example * len(data))

    print("\n--- Instruction Data Estimate ---")
    print(f"Total examples: {len(data):,}")
    print(f"Sampled: {n_sample:,} ({sample_fraction*100:.1f}%)")
    print(f"Avg tokens/example: {avg_tokens_per_example:.1f}")
    print(f"Estimated total tokens: {estimated_total_tokens:,}")

    return estimated_total_tokens



if __name__ == "__main__":




    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    num_epochs = OTHER_SETTINGS["instruction_epochs"]

    checkpoint_dir = "checkpoints_instruct"
    sample_log_path = os.path.join(checkpoint_dir, "text_sample_log.txt")

    with open(sample_log_path, "w", encoding="utf-8") as f:
        f.write(f"Training run started: {YUNA_GPT_CONFIG}\n\n")  # optional header, also clears the file

    ############LOAD DATASET HERE

    reddit = load_dataset(
    "ChaoticNeutrals/Reddit-NSFW-Writing_Prompts_ShareGPT",
    split="train")
    reddit_data = []

    for row in reddit:
        conversations = row["conversations"]

        if len(conversations) < 2:
            continue

        if conversations[0]["from"] != "human":
            continue

        if conversations[1]["from"] != "gpt":
            continue

        reddit_data.append({
            "instruction": conversations[0]["value"],
            "input": "",
            "output": conversations[1]["value"],
        })



    #POTENTIONAL DATASET REFERENCES https://huggingface.co/mradermacher/MS3.2-PaintedFantasy-v4.1-24B-ultra-uncensored-heretic-v2-i1-GGUF

    # alpaca_clean = load_dataset("yahma/alpaca-cleaned", split="train")
    # alpaca_data = [{"instruction": r["instruction"], "input": r["input"], "output": r["output"]} for r in alpaca_clean]

    # dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    # dolly_data = [{"instruction": r["instruction"], "input": r["context"], "output": r["response"]} for r in dolly]

    # longform = load_dataset("akoksal/LongForm", split="train")
    # longform_data = [{"instruction": r["input"], "input": "", "output": r["output"]} for r in longform]

    writingprompts = load_dataset("euclaise/writingprompts", split="train")
    n = int(0.2 * len(writingprompts))
    writingprompts = writingprompts.shuffle(seed=42).select(range(n))
    writingprompts_data = [
        {
            "instruction": r["prompt"].strip().removeprefix("[ WP ] ``").strip(),
            "input": "",
            "output": r["story"]
        }
        for r in writingprompts
    ]


    #data = alpaca_data + dolly_data + longform_data + writingprompts_data + reddit_data
    data = writingprompts_data + reddit_data
    random.seed(42)
    random.shuffle(data)

    save_instruction_data(data, "instruction_data.jsonl")
    print(f"Total combined examples: {len(data)}")
    print("Estimated Tokens Per Epoch ---> " + str(estimate_instruction_tokens("instruction_data.jsonl")))
    
    

    # --- load checkpoint ---
    checkpoint_path = "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    model = TransformerModel(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()


    # fine-tuning: lower LR, fewer epochs than pretraining — standard practice
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-5, weight_decay=0.1
    )
    
    
    

    tokenizer = Tokenizer.from_file("yuna_bpe.json")
    eot_id = tokenizer.token_to_id("<|endoftext|>")  # real, valid token ID
    print(f"EOT id: {eot_id}")  # sanity check — should be a small number, well


    train_portion = int(len(data) * 0.9)

    train_data = data[:train_portion]
    val_data = data[train_portion:]

    train_loader = create_instruction_dataloader(
        train_data, tokenizer,
        pad_token_id=eot_id,
        batch_size=14,
        allowed_max_length=YUNA_GPT_CONFIG["context_length"],
        device=device,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    val_loader = create_instruction_dataloader(
        val_data, tokenizer,
        pad_token_id=eot_id,
        batch_size=14,
        allowed_max_length=YUNA_GPT_CONFIG["context_length"],
        device=device,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.03 * total_steps)
    scheduler = get_linear_lr_scheduler(optimizer, warmup_steps, total_steps)

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, scheduler, device,
        num_epochs=num_epochs,
        eval_freq=100,
        eval_iter=20,
        start_context=format_input(val_data[0]),
        tokenizer=tokenizer,
        checkpoint_dir="checkpoints_instruct",
        config=checkpoint["config"],
        sample_log_path=sample_log_path,
        save_every_tokens=10000000
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")