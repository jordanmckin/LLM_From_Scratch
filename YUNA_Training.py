import tiktoken
import torch
import matplotlib.pyplot as plt
import numpy
import os
from datetime import datetime
from tokenizers import Tokenizer
from Transformer import TransformerModel, generate_text_simple
from DataLoaderV2 import create_dataloader
from YUNA_GPT import generate, text_to_token_ids, token_ids_to_text
from YUNA_CONFIG import YUNA_GPT_CONFIG, OTHER_SETTINGS

import math
from torch.optim.lr_scheduler import LambdaLR



# "Shift + Alt + F = Formatting"



def get_linear_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        return max(0.0, 1.0 - progress)  # straight-line decay to 0

    return LambdaLR(optimizer, lr_lambda)


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, max_loss=10):
    # keep only points where BOTH train and val are under the threshold,
    # so the x-axes (epochs/tokens) stay aligned with the filtered y-values
    filtered = [
        (e, t, tr, va)
        for e, t, tr, va in zip(epochs_seen, tokens_seen, train_losses, val_losses)
        if tr < max_loss and va < max_loss
    ]

    if not filtered:
        print("No points under max_loss threshold yet — skipping plot")
        return None

    epochs_f, tokens_f, train_f, val_f = zip(*filtered)

    fig, ax1 = plt.subplots()

    ax1.plot(epochs_f, train_f, label="Training loss")
    ax1.plot(epochs_f, val_f, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    ax2 = ax1.twiny()
    ax2.plot(tokens_f, train_f, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    return fig


def generate_and_print_sample(
    model,
    tokenizer,
    device,
    start_context,
    max_new_tokens=200,
    temperature=0.8,
    top_k=40,
    log_path=None,
    step_label="",
):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate(
            model=model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            temperature=temperature,
            top_k=top_k,
            tokenizer=tokenizer,
        )

    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print("Output text:\n", decoded_text)

    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as f:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"--- {step_label} | {current_time} ---\n")
            f.write(decoded_text.replace("\n", " "))
            f.write("\n\n")

    model.train()


def calc_loss_batch(input_batch, target_batch, model, device, ignore_index=-100):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten(), ignore_index=ignore_index
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None, ignore_index=-100):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device, ignore_index=ignore_index
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(
    model, train_loader, val_loader, device, eval_iter, ignore_index=-100
):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader,
            model,
            device,
            num_batches=eval_iter,
            ignore_index=ignore_index,
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter, ignore_index=ignore_index
        )
    model.train()
    return train_loss, val_loss


def get_model_state_dict(model):
    """Handles both compiled and uncompiled models consistently."""
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    return model.state_dict()


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    checkpoint_dir="checkpoints",
    config=None,
    sample_log_path=None,
    ignore_index=-100,
    save_every_tokens=200_000_000,
    start_epoch=0,
    start_global_step=-1,
    start_tokens_seen=0,
):
    train_losses, val_losses, track_tokens_seen = [], [], []

    tokens_seen = start_tokens_seen
    global_step = start_global_step
    best_val_loss = float("inf")

    # make sure the next token-based checkpoint threshold is correctly ahead of where we resumed
    next_save_tokens = ((tokens_seen // save_every_tokens) + 1) * save_every_tokens

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:

            optimizer.zero_grad()

            loss = calc_loss_batch(
                input_batch, target_batch, model, device, ignore_index=ignore_index
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # Count tokens processed
            batch_tokens = input_batch.numel()
            tokens_seen += batch_tokens

            global_step += 1

            # -------------------------------------------------
            # Evaluation
            # -------------------------------------------------

            if global_step % eval_freq == 0:

                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter,
                    ignore_index=ignore_index,
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                current_lr = scheduler.get_last_lr()[0]

                print(
                    f"Ep {epoch+1} "
                    f"(Step {global_step:06d}) "
                    f"Tokens {tokens_seen:,}: "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}, "
                    f"LR {current_lr:.2e}"
                )

                # -----------------------------
                # Latest checkpoint
                # -----------------------------

                torch.save(
                    {
                        "model_state_dict": get_model_state_dict(model),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "tokens_seen": tokens_seen,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "config": config,
                    },
                    os.path.join(checkpoint_dir, "checkpoint_latest.pt"),
                )

                # -----------------------------
                # Best checkpoint
                # -----------------------------

                if val_loss < best_val_loss:

                    best_val_loss = val_loss

                    torch.save(
                        {
                            "model_state_dict": get_model_state_dict(model),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "epoch": epoch,
                            "global_step": global_step,
                            "tokens_seen": tokens_seen,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "config": config,
                        },
                        os.path.join(checkpoint_dir, "checkpoint_best.pt"),
                    )

                    print(f"New best model saved! " f"Val loss: {val_loss:.4f}")

                # -----------------------------
                # Loss plot
                # -----------------------------

                epochs_seen_partial = torch.linspace(0, epoch + 1, len(train_losses))

                plot_losses(
                    epochs_seen_partial, track_tokens_seen, train_losses, val_losses
                )

                plt.savefig(os.path.join(checkpoint_dir, "loss_latest.pdf"))

                plt.close()

                # -----------------------------
                # Generate sample
                # -----------------------------

                generate_and_print_sample(
                    model,
                    tokenizer,
                    device,
                    start_context,
                    max_new_tokens=250,
                    temperature=0.7,
                    top_k=20,
                    log_path=sample_log_path,
                    step_label=(
                        f"Epoch {epoch+1}, "
                        f"Step {global_step:06d}, "
                        f"Tokens {tokens_seen:,}, "
                        f"Val loss {val_loss:.3f}"
                    ),
                )

            # -------------------------------------------------
            # Token based checkpoint saves
            # -------------------------------------------------

            if tokens_seen >= next_save_tokens:

                checkpoint_name = f"checkpoint_{tokens_seen//1_000_000}M_tokens.pt"

                print(f"Saving token checkpoint: {checkpoint_name}")

                torch.save(
                    {
                        "model_state_dict": get_model_state_dict(model),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "tokens_seen": tokens_seen,
                        "config": config,
                    },
                    os.path.join(checkpoint_dir, checkpoint_name),
                )

                next_save_tokens += save_every_tokens

        # -------------------------------------------------
        # End-of-epoch checkpoint
        # -------------------------------------------------

        epoch_checkpoint_name = f"checkpoint_epoch_{epoch + 1:03d}.pt"

        print(f"Saving end-of-epoch checkpoint: {epoch_checkpoint_name}")

        torch.save(
            {
                "model_state_dict": get_model_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "tokens_seen": tokens_seen,
                "config": config,
            },
            os.path.join(checkpoint_dir, epoch_checkpoint_name),
        )

    return train_losses, val_losses, track_tokens_seen


def main(config, settings=None, resume_from=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 7:
            torch.set_float32_matmul_precision("high")
            print("Tensor cores enabled")

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    sample_log_path = os.path.join(checkpoint_dir, "text_sample_log.txt")

    # don't wipe the log if we're resuming — append instead
    log_mode = "a" if resume_from else "w"
    with open(sample_log_path, log_mode, encoding="utf-8") as f:
        if resume_from:
            f.write(f"\n--- Resuming training: {config} ---\n\n")
        else:
            f.write(f"Training run started: {config}\n\n")

    model = TransformerModel(config)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        fused=True,
    )

    tokenizer = Tokenizer.from_file("yuna_bpe.json")

    train_loader = create_dataloader(
        "train.bin",
        batch_size=settings["batch_size"],
        max_length=config["context_length"],
        stride=config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader(
        "val.bin",
        batch_size=settings["batch_size"],
        max_length=config["context_length"],
        stride=config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    total_steps = len(train_loader) * settings["expected_epochs"]
    warmup_steps = int(0.03 * total_steps)
    scheduler = get_linear_lr_scheduler(optimizer, warmup_steps, total_steps)

    start_epoch = 0
    start_global_step = -1
    start_tokens_seen = 0

    if resume_from is not None:
        print(f"Resuming from {resume_from}...")
        checkpoint = torch.load(resume_from, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # start_epoch = checkpoint["epoch"]
        start_epoch = 1
        start_global_step = checkpoint["global_step"]
        start_tokens_seen = checkpoint.get("tokens_seen", 0)

        print(
            f"Resumed at epoch {start_epoch+1}, step {start_global_step}, "
            f"tokens_seen {start_tokens_seen:,}, "
            f"val_loss {checkpoint.get('val_loss', 'N/A')}"
        )

    print(f"len(train_loader): {len(train_loader)}")
    print(f"total_steps: {total_steps}")
    print(f"warmup_steps: {warmup_steps}")
    print(f"current LR: {scheduler.get_last_lr()[0]}")

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        num_epochs=settings["num_epochs"],
        eval_freq=500,
        eval_iter=5,
        start_context="Hello",
        tokenizer=tokenizer,
        config=config,
        sample_log_path=sample_log_path,
        start_epoch=start_epoch,
        start_global_step=start_global_step,
        start_tokens_seen=start_tokens_seen,
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    ###NEW
    ###NEW
    train_losses, val_losses, tokens_seen, model = main(
        YUNA_GPT_CONFIG, OTHER_SETTINGS
    )

    ###RESUME
    ###RESUME
    # Important. this starts over from the newest epoch
    # train_losses, val_losses, tokens_seen, model = main(YUNA_GPT_CONFIG, OTHER_SETTINGS, resume_from=os.path.join("checkpoints", "checkpoint_latest.pt"))

    # Plot results
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    # Save and load model
    torch.save(get_model_state_dict(model), "model.pth")  # changed
    model = TransformerModel(YUNA_GPT_CONFIG)
    model.load_state_dict(torch.load("model.pth", weights_only=True))
