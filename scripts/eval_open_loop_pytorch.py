"""Open-loop PyTorch evaluation for pi0/pi0.5 checkpoints.

This computes the same teacher-forced diffusion loss used during training on
dataset batches. It does not run LIBERO simulation and is not a success-rate
metric.
"""

import dataclasses
import logging
import pathlib
import time

import jax
import safetensors.torch
import torch
import tqdm
from transformers.modeling_utils import no_init_weights
import tyro

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.training.config as _config
import openpi.training.data_loader as _data


@dataclasses.dataclass
class Args:
    config_name: str = "pi05_libero"
    checkpoint_dir: pathlib.Path = pathlib.Path(
        "checkpoints/pi05_libero/pi05_libero_xpu_8card_input_opt/30000"
    )
    num_batches: int = 10
    batch_size: int | None = None
    num_workers: int | None = None
    device: str | None = None
    shuffle: bool = False
    seed: int = 42


def _init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname).1s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _load_model(config: _config.TrainConfig, checkpoint_dir: pathlib.Path, device: torch.device) -> torch.nn.Module:
    weight_path = checkpoint_dir / "model.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing checkpoint weights: {weight_path}")

    model_cfg = config.model
    if not isinstance(model_cfg, openpi.models.pi0_config.Pi0Config):
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    with no_init_weights():
        model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg)
    model.paligemma_with_expert.paligemma.tie_weights()
    model = model.to(device)

    logging.info("Loading checkpoint weights from %s", weight_path)
    safetensors.torch.load_model(model, weight_path, strict=True, device=str(device))
    model.eval()
    return model


def main(args: Args) -> None:
    _init_logging()

    config = _config.get_config(args.config_name)
    if args.batch_size is not None:
        config = dataclasses.replace(config, batch_size=args.batch_size)
    if args.num_workers is not None:
        config = dataclasses.replace(config, num_workers=args.num_workers)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)

    logging.info("config=%s batch_size=%d num_workers=%d", config.name, config.batch_size, config.num_workers)
    logging.info("model_config=%s", config.model)
    logging.info("device=%s num_batches=%d shuffle=%s seed=%d", device, args.num_batches, args.shuffle, args.seed)

    loader = _data.create_data_loader(
        config,
        framework="pytorch",
        shuffle=args.shuffle,
        num_batches=args.num_batches,
    )
    model = _load_model(config, args.checkpoint_dir, device)

    losses: list[float] = []
    start = time.time()
    with torch.inference_mode():
        for observation, actions in tqdm.tqdm(loader, total=args.num_batches, desc="Open-loop eval"):
            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions = actions.to(device=device, dtype=torch.float32)
            batch_losses = model(observation, actions)
            if isinstance(batch_losses, list | tuple):
                batch_losses = torch.stack(batch_losses)
            loss = batch_losses.mean()
            losses.append(float(loss.detach().cpu()))
            if len(losses) >= args.num_batches:
                break

    elapsed = time.time() - start
    mean_loss = sum(losses) / max(1, len(losses))
    logging.info("open_loop_loss=%.6f batches=%d elapsed=%.1fs sec_per_batch=%.3f", mean_loss, len(losses), elapsed, elapsed / max(1, len(losses)))


if __name__ == "__main__":
    main(tyro.cli(Args))
