#!/usr/bin/env python3
"""Utilities to fetch the tiny offline model and demo embedding bank used by the
external embedding integration examples.

The repo avoids shipping binary checkpoints to keep the history lightweight.
Run this helper to materialize the assets locally before launching
``examples/train_lora/mini_external.yaml``.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
)


def copy_selected_files(src: Path, dest: Path, filenames: Iterable[str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        src_path = src / name
        if not src_path.exists():
            raise FileNotFoundError(
                f"Required file {name!r} was not present in snapshot {src}."
            )
        shutil.copy2(src_path, dest / name)


def download_tiny_model(output_dir: Path, repo_id: str, revision: str | None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - convenience guard for optional dep
        raise SystemExit(
            "huggingface_hub is required to download the demo checkpoint. Install it "
            "before rerunning this helper."
        ) from exc

    snapshot_path = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=list(MODEL_FILES),
        )
    )
    copy_selected_files(snapshot_path, output_dir, MODEL_FILES)


def build_embedding_bank(
    output_path: Path,
    dim: int,
    count: int,
    prefix: str,
    seed: int,
) -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - convenience guard for optional dep
        raise SystemExit(
            "torch is required to author the demo embedding bank. Install it before "
            "rerunning this helper."
        ) from exc

    if count <= 0:
        raise ValueError("Number of demo ids must be positive.")
    if dim <= 0:
        raise ValueError("Embedding dimension must be positive.")

    torch.manual_seed(seed)
    tensors = {
        f"{prefix}{idx}": torch.randn(dim) for idx in range(count)
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("data/tiny-external-model"),
        help="Directory where the tiny causal LM checkpoint will be stored.",
    )
    parser.add_argument(
        "--embedding-bank",
        type=Path,
        default=Path("data/external_embedding_bank_demo.pt"),
        help="Path of the generated torch archive containing demo embeddings.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=1024,
        help="Dimensionality for each generated embedding vector.",
    )
    parser.add_argument(
        "--num-ids",
        type=int,
        default=4,
        help="How many demo embedding identifiers to author.",
    )
    parser.add_argument(
        "--id-prefix",
        type=str,
        default="demo",
        help="Prefix used when generating embedding identifiers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when sampling demo embeddings.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="hf-internal-testing/tiny-random-LlamaForCausalLM",
        help="Model repository that hosts a compact LLaMA-like checkpoint.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional revision of the model repository to pin.",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Only (re)generate the embedding bank without downloading the model.",
    )
    parser.add_argument(
        "--skip-bank",
        action="store_true",
        help="Only download the model without touching the embedding bank.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_model:
        download_tiny_model(args.model_output, args.repo_id, args.revision)
        print(f"Saved tiny checkpoint to {args.model_output.resolve()}")

    if not args.skip_bank:
        build_embedding_bank(
            args.embedding_bank,
            args.embedding_dim,
            args.num_ids,
            args.id_prefix,
            args.seed,
        )
        print(f"Wrote demo embedding bank to {args.embedding_bank.resolve()}")


if __name__ == "__main__":
    main()
