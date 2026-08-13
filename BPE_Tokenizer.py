from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tqdm import tqdm
from YUNA_CONFIG import YUNA_GPT_CONFIG

def train_bpe_tokenizer(
    input_file="shuffled_raw.txt",  # use your shuffled version specifically
    output_file="murasaki_bpe.json",
    vocab_size=YUNA_GPT_CONFIG["vocab_size"],
    chunk_size=10_000_000,
    max_chars=500_000_000  # ~500MB sample, adjust as desired
):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        min_frequency=2,
        show_progress=True,
    )

    def file_iterator():
        chars_read = 0
        with open(input_file, "r", encoding="utf-8") as f:
            while chars_read < max_chars:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                chars_read += len(chunk)
                yield chunk

    print(f"Training BPE tokenizer on ~{max_chars/1e6:.0f}M chars, target vocab_size={vocab_size}...")
    tokenizer.train_from_iterator(file_iterator(), trainer=trainer)

    tokenizer.save(output_file)
    print(f"Saved tokenizer to {output_file}")
    print(f"Actual vocab size: {tokenizer.get_vocab_size()}")

    return tokenizer

if __name__ == "__main__":
    train_bpe_tokenizer(
        input_file="shuffled_raw.txt",
        output_file="yuna_bpe.json",
        vocab_size=YUNA_GPT_CONFIG["vocab_size"]
    )