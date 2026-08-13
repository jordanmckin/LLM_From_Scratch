YUNA_GPT_CONFIG = {
    "vocab_size": 32000,
    "context_length": 1024,
    "emb_dim": 512,  # Embedding dimension 640
    "n_heads": 8,  # Number of attention heads 10
    "n_layers": 12,  # Number of layers 16
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False,  # Query-key-value bias
}

OTHER_SETTINGS = {
    "learning_rate": 5e-4,
    "num_epochs": 3,
    "instruction_epochs": 2,
    "expected_epochs": 3,
    "batch_size": 30,
    "weight_decay": 0.1,
}


def estimate_epochs(cfg, total_tokens, target_ratio=15):
    """
    Estimates a reasonable expected_epochs based on the token-to-parameter
    heuristic (aim for ~target_ratio tokens per parameter, TOTAL across all epochs).
    """
    vocab_size = cfg["vocab_size"]
    context_length = cfg["context_length"]
    emb_dim = cfg["emb_dim"]
    n_layers = cfg["n_layers"]

    token_emb = vocab_size * emb_dim  # tied weights, no separate out_head
    pos_emb = context_length * emb_dim
    blocks = n_layers * 12 * emb_dim**2
    total_params = token_emb + pos_emb + blocks

    target_total_tokens_seen = total_params * target_ratio
    epochs = target_total_tokens_seen / total_tokens

    print(f"Estimated params: {total_params/1e6:.1f}M")
    print(f"Dataset size: {total_tokens/1e6:.0f}M tokens")
    print(
        f"Target total tokens seen (ratio={target_ratio}x): {target_total_tokens_seen/1e6:.0f}M"
    )
    print(f"Suggested expected_epochs: {epochs:.2f}")

    return epochs


def estimate_ratio_and_recommendation(cfg, actual_input_tokens):
    """
    Given your actual token count (accounting for however many epochs
    you actually plan to run), tells you your real tokens-per-parameter
    ratio and flags whether it's in a healthy range.
    """
    vocab_size = cfg["vocab_size"]
    context_length = cfg["context_length"]
    emb_dim = cfg["emb_dim"]
    n_layers = cfg["n_layers"]

    token_emb = vocab_size * emb_dim
    pos_emb = context_length * emb_dim
    blocks = n_layers * 12 * emb_dim**2
    total_params = token_emb + pos_emb + blocks

    ratio = actual_input_tokens / total_params

    print(f"Estimated params: {total_params/1e6:.1f}M")
    print(f"Actual total tokens processed: {actual_input_tokens/1e6:.0f}M")
    print(f"Tokens-per-parameter ratio: {ratio:.1f}x")

    if ratio < 5:
        verdict = "Likely UNDER-trained — model has more capacity than data justifies. More data or fewer epochs of headroom available."
    elif ratio < 10:
        verdict = "On the lean side — workable, but you're below the classic 10-20x comfort zone."
    elif ratio <= 20:
        verdict = "Healthy — solidly within the commonly-used 10-20x range."
    elif ratio <= 40:
        verdict = "Generous — more than the classic heuristic calls for, likely fine, diminishing returns setting in."
    else:
        verdict = "Likely MORE than needed for this model size — you may be past the point where more tokens/epochs meaningfully help, given this capacity. Consider whether a bigger model would use this data better, or whether repetition/epochs are adding more overfitting risk than benefit."

    print(f"Verdict: {verdict}")

    return ratio


# usage — pass in your REAL total tokens processed, i.e. tokens_in_dataset * epochs_actually_run
def estimate_params(cfg, tied_weights=True):
    vocab_size = cfg["vocab_size"]
    context_length = cfg["context_length"]
    emb_dim = cfg["emb_dim"]
    n_layers = cfg["n_layers"]

    token_emb = vocab_size * emb_dim
    pos_emb = context_length * emb_dim
    out_head = 0 if tied_weights else vocab_size * emb_dim

    # per block: 4*d^2 (attention QKV+proj) + 8*d^2 (feedforward 4x expand/compress) = 12*d^2
    per_block = 12 * emb_dim**2
    blocks_total = n_layers * per_block

    total = token_emb + pos_emb + out_head + blocks_total

    print(
        f"Token embedding{' (tied w/ out_head)' if tied_weights else ''}: {token_emb/1e6:.2f}M"
    )
    print(f"Positional embedding: {pos_emb/1e6:.2f}M")
    if not tied_weights:
        print(f"Output head (separate): {out_head/1e6:.2f}M")
    print(f"Transformer blocks ({n_layers} layers): {blocks_total/1e6:.2f}M")
    print(f"Total: {total/1e6:.2f}M params")

    return total


if __name__ == "__main__":
    estimate_ratio_and_recommendation(
        YUNA_GPT_CONFIG, actual_input_tokens=2_100_000_000 * 2
    )
    estimate_params(YUNA_GPT_CONFIG, tied_weights=True)
    estimate_epochs(YUNA_GPT_CONFIG, total_tokens=1_100_000_000)
    estimate_ratio_and_recommendation(
        YUNA_GPT_CONFIG, actual_input_tokens=1_100_000_000 * 1
    )
