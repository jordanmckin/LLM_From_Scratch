"""Offline smoke test for tokenizer, model save/reload, and generation."""

import gc
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


def make_tokenizer(path):
    backend = Tokenizer(models.BPE(unk_token="<|unk|>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=64,
        special_tokens=["<|begin_of_text|>", "<|end_of_text|>", "<|unk|>", "<|pad|>"],
    )
    backend.train_from_iterator(
        ["Yuna writes a small test.", "A second sentence tests generation."],
        trainer=trainer,
    )
    backend.save(str(path / "tokenizer.json"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"),
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
    )
    tokenizer.save_pretrained(path)


def main():
    torch.manual_seed(7)
    with tempfile.TemporaryDirectory(prefix="yuna_llama_smoke_") as temp:
        root = Path(temp)
        make_tokenizer(root)
        tokenizer = AutoTokenizer.from_pretrained(root)
        config = LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=32,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            tie_word_embeddings=True,
        )
        model = LlamaForCausalLM(config).eval()
        batch = tokenizer("Yuna writes", return_tensors="pt")
        with torch.no_grad():
            before = model(**batch).logits
        model.save_pretrained(root / "model", safe_serialization=True)
        tokenizer.save_pretrained(root / "model")
        reloaded = AutoModelForCausalLM.from_pretrained(root / "model").eval()
        with torch.no_grad():
            after = reloaded(**batch).logits
        assert before.shape[-1] == len(tokenizer)
        assert torch.allclose(before, after, atol=1e-5, rtol=1e-5)
        generated = reloaded.generate(**batch, max_new_tokens=3, do_sample=False)
        assert generated.shape[1] > batch["input_ids"].shape[1]
        # Transformers keeps the safetensors memory map alive until the model
        # is released.  Explicit cleanup avoids a Windows temp-directory lock.
        del reloaded, model
        gc.collect()
    print("LLaMA HF smoke test passed")


if __name__ == "__main__":
    main()
