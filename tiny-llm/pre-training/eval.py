#!/usr/bin/env python3
from __future__ import annotations

from train import *

def prepare_valid_cache(
    *,
    valid_text_path: Path,
    cache_dir: Path,
    tokenizer: Any,
    overwrite: bool,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    valid_bin = cache_dir / "valid_tokens.bin"
    if not valid_bin.exists() or overwrite:
        if not valid_text_path.exists():
            raise FileNotFoundError(
                f"Missing {valid_text_path}. Download data first, for example:\n"
                "python data/download_tinystories.py subset --target-train-tokens 50000000"
            )
        encode_to_bin(valid_text_path, valid_bin, tokenizer)
    return valid_bin

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained GPT on validation loss.")
    parser.add_argument("config", type=Path, help="Path to a JSON training config.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    model_config = GPTConfig(**config["model"])
    training_config = config["training"]

    set_seed(config.get("seed", 1337))
    device = select_device(training_config["device"])
    dtype_name = training_config.get("dtype", "float32")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    if device == "cpu" or dtype == torch.float32:
        ctx = nullcontext()
    else:
        ctx = torch.autocast(device_type=device, dtype=dtype)

    tokenizer = load_tokenizer(resolve_path(data_config["tokenizer_path"]))
    valid_bin = prepare_valid_cache(
        valid_text_path=resolve_path(data_config["valid_text_path"]),
        cache_dir=resolve_path(data_config["cache_dir"]),
        tokenizer=tokenizer,
        overwrite=data_config["overwrite_cache"],
    )
    valid_data = load_memmap(valid_bin)

    checkpoint = torch.load(Path(resolve_path(training_config["out_dir"])) / "best.pt", map_location=device)
    model = GPT(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"run_name: {config['run_name']}")
    print(f"device: {device}")
    print(f"parameters: {model.parameter_count() / 1_000_000:.2f}M")
    print(f"valid tokens: {len(valid_data):,}")

    eval_iters = max(training_config["eval_iters"], 1024)
    batch_size = int(training_config["batch_size"])
    block_size = model_config.block_size


    losses = torch.zeros(eval_iters)
    for index in range(eval_iters):
        x, y = get_batch(valid_data, batch_size=batch_size, block_size=block_size, device=device)
        with ctx:
            _, loss = model(x, y)
        losses[index] = loss.item()


    print(
        f"valid {losses.mean().item():.4f}"
    )



if __name__ == "__main__":
    main()